import yaml
import wandb
import torch
import os
import time
import logging
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import warnings

from dataload.data_load_seq2_fea import ABAWData, collate_fn
warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "dryrun"

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.combine_fea_model import MTLModel
from utils import AverageMeter, MTLLoss, ProgressMeter, build_optimizer_and_scheduler, final_evaluation, setup_logging, setup_seed

def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"{name} contains NaN or Inf")

def get_total_params(optimized_parameters):
    total_params = 0
    for param_group in optimized_parameters:
        params = param_group['params']
        total_params += sum(p.numel() for p in params if p.requires_grad)
    return total_params

def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        config = yaml.safe_load(file)
    return config

def main_worker(gpu, ngpus_per_node, config):
    config["gpu"] = gpu
    config["rank"] = config["node_rank"] * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=config["world_size"], rank=config["rank"])

    if config["rank"] == 0:
        wandb.init(
            project=f"ABAW_MTL",
            name=config["wandb_name"],
            config=config,
            tags=["DDP implementation"],
        )

    setup_seed(config["seed"])
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    log_dir = os.path.join(config["log_dir"], f'{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

    setup_logging(filename=os.path.join(log_dir, 'log.txt'))
    logger = logging.getLogger(__name__)
    if config["rank"] == 0:
        logger.info(f"==> Arguments: {config}")

    model = MTLModel(config).to(device)

    optimized_parameters = [
        {'params': filter(lambda p: p.requires_grad, model.model.parameters()), 'lr': float(config['MAE_init_lr'])},
        {'params': filter(lambda p: p.requires_grad, model.feat_fusion.parameters()), 'lr': float(config['temporal_lr'])},
        {'params': filter(lambda p: p.requires_grad, model.temporal_convergence.parameters()), 'lr': float(config['temporal_lr'])} if config['time_model'] else None,
        {'params': filter(lambda p: p.requires_grad, model.AUhead.parameters()), 'lr': float(config['head_lr'])} if config['AU'] else None,
        {'params': filter(lambda p: p.requires_grad, model.ExprHead.parameters()), 'lr': float(config['head_lr'])} if config['EXPR'] else None,
        {'params': filter(lambda p: p.requires_grad, model.vhead.parameters()), 'lr': float(config['head_lr'])} if config['VA_Valence'] else None,
        {'params': filter(lambda p: p.requires_grad, model.ahead.parameters()), 'lr': float(config['head_lr'])} if config['VA_Arousal'] else None]
    
    optimized_parameters = [param for param in optimized_parameters if param is not None]
    optimizer, scheduler = build_optimizer_and_scheduler(config, optimized_parameters)

    if config["rank"] == 0:
        print(model)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    if config["resume_model"]:
        model.load_state_dict(torch.load(config["resume_model"]), strict=False)
        if config["rank"] == 0:
            logger.info(f'==> Using Pretrain Model: {config["resume_model"]}')

    if config["rank"] == 0:
        total_params = get_total_params(optimized_parameters)
        logger.info(f"==> Total params: {total_params / 1e6:.2f}M")

        wandb.config.update({"log_dir": log_dir}, allow_val_change=True)
        wandb.summary["model_structure"] = model

    config["best_metric"] = float("-inf")

    train_set = ABAWData(config, is_train=True)
    val_set = ABAWData(config, is_train=False)

    train_sampler = DistributedSampler(train_set, num_replicas=config["world_size"], rank=config["rank"])
    val_sampler = DistributedSampler(val_set, num_replicas=config["world_size"], rank=config["rank"])
    train_loader = DataLoader(train_set, batch_size=config["batchsize"], collate_fn=collate_fn, num_workers=config["num_workers"], sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=config["batchsize"], collate_fn=collate_fn, num_workers=config["num_workers"], sampler=val_sampler)

    if config["rank"] == 0:
        logger.info(f"****************     Starting training      *********************")
        logger.info(f"The amount of training data: {len(train_set)}, val_set data: {len(val_set)}")

    for epoch in range(config["epoch"]):
        train_sampler.set_epoch(epoch)
       
        metric = evaluation(val_loader, model, epoch, config, device, logger)
        train(train_loader, model, optimizer, epoch, config, device, logger)
        
        scheduler.step()
        if config["rank"] == 0 and epoch < 50:
            saving_checkpoint(model, epoch, config, log_dir, logger, metric)
    gc.collect()

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def train(train_loader, model, optimizer, epoch, config, device, logger):
    if config["rank"] == 0:
        logger.info(f'*******************   Training at epoch {epoch}   ***********************')
    torch.set_grad_enabled(True)
    model.train()

    meters = []

    if config["AU"]:
        au_loss_mtr = AverageMeter('AU_Loss', ":.5f")
        meters.append(au_loss_mtr)
    if config["EXPR"]:
        expr_loss_mtr = AverageMeter('EXPR_Loss', ":.5f")
        meters.append(expr_loss_mtr)
    if config["VA_Arousal"]:
        arousal_loss_mtr = AverageMeter('Arousal_Loss', ":.5f")
        meters.append(arousal_loss_mtr)
    if config["VA_Valence"]:
        valence_loss_mtr = AverageMeter('Valence_Loss', ":.5f")
        meters.append(valence_loss_mtr)

    progress = ProgressMeter(
        len(train_loader),
        meters,
        prefix="Epoch: [{}]".format(epoch),)

    mtl_lsfuns = MTLLoss()

    for i, batchdata in enumerate(train_loader):
        images, valence, arousal, expression, aus, _, au_fea_tensor, expr_fea_tensor, v_fea_tensor = batchdata
        if len(images.shape) > 4:
            batchsize, sequence_len, _, _, _ = images.shape
            data_num = batchsize * sequence_len
        else:
            batchsize, _, _, _ = images.shape
            data_num = batchsize

        valence = valence.view(data_num).to(device)
        arousal = arousal.view(data_num).to(device)
        expression = expression.view(data_num).to(device)
        aus = aus.view(data_num, -1).to(device)

        if au_fea_tensor is not None:
            au_fea_tensor = au_fea_tensor.to(device)
        if expr_fea_tensor is not None:
            expr_fea_tensor = expr_fea_tensor.to(device)
        if v_fea_tensor is not None:
            v_fea_tensor = v_fea_tensor.to(device)
            
        images_tensor = images.to(device)

        au_out, expr_out, vout, aout = model(images_tensor, au_fea_tensor, expr_fea_tensor, v_fea_tensor)
        total_loss = 0

        if config["AU"]:
            au_mask = (aus != -1).float() # mask
            au_loss = mtl_lsfuns.au_bce_loss(au_out, aus, au_mask)
            total_loss += au_loss
            au_loss_mtr.update(reduce_tensor(au_loss.mean(), config["world_size"]).item())

        if config["EXPR"]:
            expression_mask = (expression != -1).float()
            expr_loss = mtl_lsfuns.expr_ce_loss(expr_out, expression, expression_mask)
            expr_loss_mtr.update(reduce_tensor(expr_loss.mean(), config["world_size"]).item())
            total_loss += expr_loss

        if config["VA_Arousal"]:
            arousal_mask = (arousal != -5).float()
            aout = aout.squeeze()
            a_ccc_loss = mtl_lsfuns.ccc_loss(arousal, aout, arousal_mask)
            total_loss += a_ccc_loss
            arousal_loss_mtr.update(reduce_tensor(a_ccc_loss.mean(), config["world_size"]).item())

        if config["VA_Valence"]:
            valence_mask = (valence != -5).float()
            vout = vout.squeeze()
            v_ccc_loss = mtl_lsfuns.ccc_loss(valence, vout, valence_mask)
            total_loss += v_ccc_loss

            valence_loss_mtr.update(reduce_tensor(v_ccc_loss.mean(), config["world_size"]).item())
        
        if torch.isnan(total_loss):
            print(f"NaN loss detected at epoch {epoch}, batch {i}. Skipping this batch.")
            continue  # 跳过当前批次

        optimizer.zero_grad()
        total_loss.mean().backward()
        optimizer.step()

        if config["rank"] == 0:
            logger.info(progress.display(i))

    if config["world_size"] > 1:
        for meter in meters:
            meter.avg = reduce_tensor(torch.tensor(meter.avg).to(device), config["world_size"]).item()
        total_loss = reduce_tensor(total_loss, config["world_size"])

    if config["rank"] == 0:
        wandb.log({f"train_{mtr.name}": mtr.avg for mtr in meters})
        wandb.log({"train_total_loss": total_loss.item()})

    gc.collect()

def evaluation(val_loader, model, epoch, config, device, logger):

    if config["rank"] == 0:
        logger.info(f'*******************   Evaluation at epoch {epoch}   ***********************')
    torch.set_grad_enabled(False)
    model.eval()

    meters = []
    mtl_lsfuns = MTLLoss()
    total_loss = 0
    metric = 0

    frame_pred_set = set()

    if config["AU"]:
        au_loss_mtr = AverageMeter('Val_AU_Loss', ":.5f")
        meters.append(au_loss_mtr)
    if config["EXPR"]:
        expr_loss_mtr = AverageMeter('Val_EXPR_Loss', ":.5f")
        meters.append(expr_loss_mtr)
    if config["VA_Arousal"]:
        arousal_loss_mtr = AverageMeter('Val_Arousal_Loss', ":.5f")
        meters.append(arousal_loss_mtr)
    if config["VA_Valence"]:
        valence_loss_mtr = AverageMeter('Val_Valence_Loss', ":.5f")
        meters.append(valence_loss_mtr)
 
    for _, batchdata in enumerate(val_loader):
        images, valence, arousal, expression, aus, image_names, au_fea_tensor, expr_fea_tensor, v_fea_tensor = batchdata
        if len(images.shape) > 4:
            batchsize, sequence_len, _, _, _ = images.shape     
            data_num = batchsize * sequence_len
        else:
            batchsize, _, _, _ = images.shape   
            data_num = batchsize 
    
        images_tensor = images.to(device)

        if au_fea_tensor is not None:
            au_fea_tensor = au_fea_tensor.to(device)
        if expr_fea_tensor is not None:
            expr_fea_tensor = expr_fea_tensor.to(device)
        if v_fea_tensor is not None:
            v_fea_tensor = v_fea_tensor.to(device)

        au_out, expr_out, vout, aout = model(images_tensor, au_fea_tensor, expr_fea_tensor, v_fea_tensor)

        new_images_name = [name for img_name in image_names for name in img_name ]

        if config["AU"]:
            aus = aus.view(data_num, -1).to(device)
            au_mask = (aus != -1).float() # mask
            au_loss = mtl_lsfuns.au_bce_loss(au_out, aus, au_mask)
            total_loss += config["AU_weight"] * au_loss
            au_loss_mtr.update(reduce_tensor(au_loss.mean(), config["world_size"]).item())

            # pred
            au_pred = torch.sigmoid(au_out)
        else:
            au_pred = torch.full((data_num, 12), -1)
        
        if config["EXPR"]:
            expression = expression.view(data_num).to(device)
            expression_mask = (expression != -1).float()
            expr_loss = mtl_lsfuns.expr_ce_loss(expr_out, expression, expression_mask)
            expr_loss_mtr.update(reduce_tensor(expr_loss.mean(), config["world_size"]).item())
            total_loss += config["EXPR_weight"] * expr_loss

            # pred
            expr_pred = F.softmax(expr_out, dim=-1) 
        else:
            expr_pred = torch.full((data_num, 8), -1)
        
        if config["VA_Arousal"]:
            arousal = arousal.view(data_num).to(device)
            arousal_mask = (arousal != -5).float()
            aout = aout.squeeze(-1)
            a_ccc_loss = mtl_lsfuns.ccc_loss(arousal, aout, arousal_mask)
            total_loss += config["VA_Arousal_weight"] * a_ccc_loss
            arousal_loss_mtr.update(reduce_tensor(a_ccc_loss.mean(), config["world_size"]).item())
        else:
            aout = torch.full((data_num,), -5)

        if config["VA_Valence"]:
            valence = valence.view(data_num).to(device)
            valence_mask = (valence != -5).float()
            vout = vout.squeeze(-1)
            v_ccc_loss = mtl_lsfuns.ccc_loss(valence, vout, valence_mask)
            total_loss += config["VA_Valence_weight"] * v_ccc_loss
            valence_loss_mtr.update(reduce_tensor(v_ccc_loss.mean(), config["world_size"]).item())
        else:
            vout = torch.full((data_num,), -5)

        v_pred = vout.detach().cpu().numpy()
        a_pred = aout.detach().cpu().numpy()
        valence = valence.detach().cpu().numpy()
        arousal = arousal.detach().cpu().numpy()

        expr_pred = expr_pred.detach().cpu().numpy()
        expression = expression.detach().cpu().numpy()

        aus = aus.detach().cpu().numpy()
        au_pred = au_pred.detach().cpu().numpy()

        for k in range(data_num):
            frame_pred_set.add((
                new_images_name[k], 
                v_pred[k],
                a_pred[k],
                *expr_pred[k],
                *au_pred[k]))
    
    frame_pred_list = list(frame_pred_set)
    all_frame_pred_lists = [None] * config["world_size"]
    dist.all_gather_object(all_frame_pred_lists, frame_pred_list)

    if config["rank"] == 0:
        combined_dict = {}
        for pred_list in all_frame_pred_lists:
            for pred in pred_list:
                image_name = pred[0]
                if image_name not in combined_dict:
                    combined_dict[image_name] = pred
        frame_pred_list = list(combined_dict.values())
    
    if config["world_size"] > 1:
        for meter in meters:
            meter.avg = reduce_tensor(torch.tensor(meter.avg).to(device), config["world_size"]).item()
        total_loss = reduce_tensor(total_loss, config["world_size"])
            
    if config["rank"] == 0:
        output_file = config["log_dir"] + f"/{config['wandb_name']}_{epoch}.txt"
        with open(output_file, 'w') as f:
            for row in frame_pred_list:
                f.write(",".join(map(str, row)) + "\n")
        logger.info(f"Save to {output_file}")
        au_F1_mean, exp_f1score, ccc_arousal, ccc_valence = final_evaluation(output_file, prob=True, config=config)  

        wandb.log({f"{mtr.name}": mtr.avg for mtr in meters})
        wandb.log({"val_total_loss": total_loss.item()})

        if config["AU"]:
            logger.info(f"Au score: {au_F1_mean}")
            wandb.log({"Au score": au_F1_mean})
            metric += au_F1_mean

        if config["EXPR"]:
            logger.info(f"EXPR score: {exp_f1score}")
            wandb.log({"EXPR score": exp_f1score})
            metric += exp_f1score

        if config["VA_Arousal"]:
            logger.info(f"Arousal score: {ccc_arousal}")
            wandb.log({"Arousal score": ccc_arousal}) 
            metric += ccc_arousal

        if config["VA_Valence"]:
            logger.info(f"Valence score: {ccc_valence}")
            wandb.log({"Valence score": ccc_valence}) 
            metric += ccc_valence
        
    del frame_pred_set, frame_pred_list, all_frame_pred_lists, au_out, expr_out, vout, aout
    gc.collect() 
        
    return metric

def saving_checkpoint(model, epoch, config, log_dir, logger, cur_metric):
    logger.info(f'Saving checkpoint at epoch {epoch}')
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'
    
    prefix = []
    if config["AU"]:
        prefix.append("AU")
    if config["EXPR"]:
        prefix.append("EXPR")
    if config["VA_Arousal"]:
        prefix.append("VA_Arousal")
    if config["VA_Valence"]:
        prefix.append("VA_Valence")

    if config["rank"] == 0:
        torch.save(model.state_dict(), f'{log_dir}/{"_".join(prefix)}_epoch{epoch}_{suffix_latest}')

        if cur_metric > config["best_metric"]:
            config["best_metric"] = cur_metric

            wandb.run.summary[f"best_metric"] = cur_metric
            torch.save(model.state_dict(), f'{log_dir}/{"_".join(prefix)}_epoch{epoch}_{suffix_best}')
            logger.info(f"Saving the best checkpoint at epoch {epoch}")
            logger.info(f"Best metric is {config['best_metric']} at epoch {epoch}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--config_file", type=str, default="/root/code/ABAW/configs/Track1_enhance_AU_Fea.yml")
    args = parser.parse_args()

    config_file = args.config_file
    config = read_yaml_to_dict(config_file)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12364'
    ngpus_per_node = torch.cuda.device_count()
    config["world_size"] = ngpus_per_node
    config["node_rank"] = 0  # For single-node training

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

if __name__ == "__main__":
    main()
