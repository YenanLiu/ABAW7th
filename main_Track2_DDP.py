import yaml
import wandb
import numpy as np
import torch
import os
import time
import logging
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataload.data_loader_frame import ABAW_Track2_Multi_Label_Frame, ABAW_Track2_single_Label_Frame
from models.ce_model import CEModel
from torch.utils.data import ConcatDataset, Subset
from utils import AverageMeter, CER_Evaluator, CERLoss, ProgressMeter, build_optimizer_and_scheduler, setup_logging, setup_seed

def get_model_ema(model, ema_ratio=1e-3):
    def ema_func(avg_param, param, num_avg):
        return (1 - ema_ratio) * avg_param + ema_ratio * param
    return torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_func)

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

def get_subset(dataset, fraction, seed):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(fraction * dataset_size))

    np.random.seed(seed)
    np.random.shuffle(indices)

    subset_indices = indices[:split]
    subset = Subset(dataset, subset_indices)
    return subset

def main_worker(gpu, ngpus_per_node, config):
    config["gpu"] = gpu
    config["rank"] = config["node_rank"] * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=config["world_size"], rank=config["rank"])

    if config["rank"] == 0:
        wandb.init(
            project=f"ABAW_CER",
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

    model = CEModel(config).to(device)

    optimized_parameters = [
        {'params': filter(lambda p: p.requires_grad, model.model.parameters()), 'lr': float(config['MAE_init_lr'])} if config['fintune_MAE'] else None,
        {'params': filter(lambda p: p.requires_grad, model.pred_head.parameters()), 'lr': float(config['pred_head_lr'])}]
    
    optimized_parameters = [param for param in optimized_parameters if param is not None]
    optimizer, scheduler = build_optimizer_and_scheduler(config, optimized_parameters)

    if config["rank"] == 0:
        print(model)
    model = DDP(model, device_ids=[gpu])

    if config["ema"]:
        ema_model = get_model_ema(model, ema_ratio=1e-3)

    if config["resume_model"]:
        model.load_state_dict(torch.load(config["resume_model"]), strict=False)
        if config["rank"] == 0:
            logger.info(f'==> Using Pretrain Model: {config["resume_model"]}')
            # logger.info(model)

    if config["rank"] == 0:
        total_params = get_total_params(optimized_parameters)
        logger.info(f"==> Total params: {total_params / 1e6:.2f}M")

        wandb.config.update({"log_dir": log_dir}, allow_val_change=True)
        wandb.summary["model_structure"] = model

    config["best_metric"] = float("-inf")
    
    if config["Data_RAF-DB-Single"] or config["Data_Aff-wild2"]:
        train_single_label_dataset = ABAW_Track2_single_Label_Frame(config, is_train=True)

    if config["Data_RAF-DB-Multi"] or config["Data_Competition"]:
        train_multi_label_dataset = ABAW_Track2_Multi_Label_Frame(config)

    val_dataset = ABAW_Track2_single_Label_Frame(config, is_train=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=config["world_size"], rank=config["rank"])
    val_loader = DataLoader(val_dataset, batch_size=config["batchsize"], num_workers=config["num_workers"], sampler=val_sampler)

    if config["rank"] == 0:
        logger.info(f"****************     Starting training      *********************")
        if config["Data_RAF-DB-Multi"] or config["Data_Competition"]:
            logger.info(f"train_multi_label_sampler data: {len(train_multi_label_dataset)}")
        if config["Data_RAF-DB-Single"] or config["Data_Aff-wild2"]:
            logger.info(f"train_single_label_sampler data: {len(train_single_label_dataset)}")
        logger.info(f"The amount of val data: {len(val_dataset)}")

    global_epoch = 0
    for stage in range(len(config["train_epoch"])):
        if len(config["train_epoch"]) == 1:
            datset_list = []
            if config["Data_RAF-DB-Single"] or config["Data_Aff-wild2"]:
                datset_list.append(train_single_label_dataset)
            if config["Data_RAF-DB-Multi"] or config["Data_Competition"]:
                datset_list.append(train_multi_label_dataset)
            train_dataset = ConcatDataset(datset_list)
        
        else: 
            multi_data_proportion = config["m_label"][stage]
            if multi_data_proportion > 0:
                sub_train_multi_label_dataset = get_subset(train_multi_label_dataset, multi_data_proportion, config["seed"])
                train_dataset = ConcatDataset([train_single_label_dataset, sub_train_multi_label_dataset])
            else:
                train_dataset = train_single_label_dataset

        train_sampler = DistributedSampler(train_dataset, num_replicas=config["world_size"], rank=config["rank"])
        train_loader = DataLoader(train_dataset, batch_size=config["batchsize"], num_workers=config["num_workers"], sampler=train_sampler)

        for _ in range(config["train_epoch"][stage]):
            train_sampler.set_epoch(global_epoch)
            train(train_loader, model, optimizer, global_epoch, config, device, logger, stage)

            if config["ema"]:
                ema_model.update_parameters(model)
                metric = evaluation(val_loader, ema_model, global_epoch, config, device, logger, stage)
            else:
                metric = evaluation(val_loader, model, global_epoch, config, device, logger, stage)
            
            global_epoch += 1
            scheduler.step()

            if config["rank"] == 0:
                if config["ema"]:
                    saving_checkpoint(ema_model, global_epoch, config, log_dir, logger, metric, suffix='ema')
                else:
                    saving_checkpoint(model, global_epoch, config, log_dir, logger, metric)

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def train(train_loader, model, optimizer, epoch, config, device, logger, stage):
    if config["rank"] == 0:
        logger.info(f'*******************   Training at epoch {epoch}   ***********************')
    torch.set_grad_enabled(True)
    model.train()

    meters = []
    progress = ProgressMeter(
        len(train_loader),
        meters,
        prefix=f"Stage:[{stage}] Epoch: [{epoch}]",
    )

    cer_lsfuns = CERLoss(config)
    if config["loss_type"] == "bce":
        loss_mtr = AverageMeter('BCE_Loss', ":.5f")
    if config["loss_type"] == "bmc":
        loss_mtr = AverageMeter('BMC_Loss', ":.5f")
    
    meters.append(loss_mtr)

    for i, batchdata in enumerate(train_loader):
        images, labels = batchdata
        images_tensor = images.to(device)
        labels_tensor = labels.to(device)

        expr_out = model(images_tensor)
        total_loss = 0
        if config["loss_type"] == "bce":
            loss = cer_lsfuns.bce_loss(expr_out.float(), labels_tensor.float())
        if config["loss_type"] == "bmc":
            loss = cer_lsfuns.bmc_loss(expr_out.float(), labels_tensor.float())
        total_loss += loss
        loss_mtr.update(reduce_tensor(loss.mean(), config["world_size"]).item())

        optimizer.zero_grad()
        loss.mean().backward()
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

def evaluation(val_loader, model, epoch, config, device, logger, stage):
    if config["rank"] == 0:
        logger.info(f'*******************   Evaluation at epoch {epoch}   ***********************')
    torch.set_grad_enabled(False)
    model.eval()

    meters = []
    progress = ProgressMeter(
        len(val_loader),
        meters,
        prefix=f"Stage:[{stage}] Epoch: [{epoch}]",
    )
    cer_eval = CER_Evaluator()

    f1_score_mtr = AverageMeter('F1_score', ":.5f")
    meters.append(f1_score_mtr)

    metric = 0

    for i, batchdata in enumerate(val_loader):
        images, labels = batchdata
        images_tensor = images.to(device)
        labels_tensor = labels.to(device)

        expr_out = model(images_tensor)
        
        f1 = cer_eval.eval(expr_out, labels_tensor)
        f1_score_mtr.update(reduce_tensor(torch.tensor(f1).to(device), config["world_size"]).item())
        
        if config["rank"] == 0:
            logger.info(progress.display(i))

    metric = f1_score_mtr.avg

    if config["world_size"] > 1:
        for meter in meters:
            meter.avg = reduce_tensor(torch.tensor(meter.avg).to(device), config["world_size"]).item()

    if config["rank"] == 0:
        wandb.log({f"{mtr.name}": mtr.avg for mtr in meters})
    return metric

def saving_checkpoint(model, epoch, config, log_dir, logger, cur_metric, suffix=None):
    logger.info(f'Saving checkpoint at epoch {epoch}')
    suffix_latest = 'latest.pth' if suffix is None else f"{suffix}_latest.pth"
    suffix_best = 'best.pth' if suffix is None else f"{suffix}_best.pth"

    if config["rank"] == 0:
        torch.save(model.state_dict(), f'{log_dir}/epoch{epoch}_{suffix_latest}')

        if cur_metric > config["best_metric"]:
            config["best_metric"] = cur_metric

            wandb.run.summary[f"best_metric"] = cur_metric
            torch.save(model.state_dict(), f'{log_dir}/epoch{epoch}_{suffix_best}')
            logger.info(f"Saving the best checkpoint at epoch {epoch}")
            logger.info(f"Best metric is {config['best_metric']} at epoch {epoch}")

def main():
    config_file = "/root/code/ABAW/configs/Track2_settings.yml"
    config = read_yaml_to_dict(config_file)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    ngpus_per_node = torch.cuda.device_count()
    config["world_size"] = ngpus_per_node
    config["node_rank"] = 0  # For single-node training

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

if __name__ == "__main__":
    main()
