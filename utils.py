import torch
import math
import logging
import random
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from torch.optim import *
from torch.distributions import MultivariateNormal as MVN
import yaml

def setup_logging(filename, resume=False):
    root_logger = logging.getLogger()

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=filename, mode='a' if resume else 'w')

    root_logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = ' '.join(entries)
        # print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')
        return msg

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

################################ Optimizer and Scheduler ########################################
    
def build_optimizer_and_scheduler(config, optimized_parameters):
    if config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(optimized_parameters, betas=(0.9, 0.999), weight_decay=config['adamW_weight_decay'])
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(optimized_parameters, betas=(0.9, 0.999))
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(optimized_parameters, momentum=float(config["momentum"]),
                          weight_decay=config["sgd_weight_decay"])

    if config["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=float(config["steplr_size"]), gamma=float(config["steplr_gamma"]))
    elif config["scheduler"] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=float(config["Msteplr_size"]), gamma=float(config["Msteplr_gamma"]))
    elif config["scheduler"] == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(config["ExponentialLR_gamma"]))
    elif config["scheduler"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=float(config["CosineAnnealingLR_T_max"]), eta_min=float(config["CosineAnnealingLR_eta_min"]))

    return optimizer, scheduler

################################ Recorder ########################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

################################ Evaluator ########################################
class MTLEvatuator:
    def ccc_cal(self, x, y, mask=None):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            x = x[mask]
            y = y[mask]

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        covariance = np.mean((x - mean_x) * (y - mean_y))

        ccc = (2 * covariance) / (var_x + var_y + (mean_x - mean_y)**2 + 1e-8)
        return ccc
    
    # (v_gt_np, v_pred_np, a_gt_np, a_pred_np)
    def mean_ccc(self, valence_true, valence_pred, arousal_true, arousal_pred, v_mask=None, a_mask=None):
        ccc_valence = self.ccc_cal(valence_true, valence_pred, v_mask)
        ccc_arousal = self.ccc_cal(arousal_true, arousal_pred, a_mask)
        mean_ccc = (ccc_valence + ccc_arousal) / 2
        return mean_ccc, ccc_valence, ccc_arousal
    
    def exp_f1(self, expression_gt, expression_prob, mask=None, prob=True): # (26666, 1) (26666, 8) 
        if prob:
            _, expression_pred = torch.max(torch.tensor(expression_prob), -1)  # torch.Size([26666])
        else:
            expression_pred = expression_prob
        if isinstance(expression_gt, torch.Tensor):
            expression_gt = expression_gt.cpu().numpy()
        if isinstance(expression_pred, torch.Tensor):
            if prob:
                expression_pred = expression_pred.cpu().numpy().reshape(expression_gt.shape[0], 1)
            else:
                expression_pred = expression_pred.cpu().numpy()

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
        
            valid_indices = mask == 1
            expression_gt = expression_gt[valid_indices]
            expression_pred = expression_pred[valid_indices]

        return f1_score(expression_gt, expression_pred, average='macro', zero_division=0)
    
    def compute_AU_F1(self, pred, label):
        pred = np.array(pred.detach().cpu())
        label = np.array(label.detach().cpu())
        AU_targets = [[] for _ in range(12)]
        AU_preds = [[] for _ in range(12)]
        F1s = []
        for i in range(pred.shape[0]):
            for j in range(12):
                p = pred[i,j]
                if p>=0.5:
                    AU_preds[j].append(1)
                else:
                    AU_preds[j].append(0)
                AU_targets[j].append(label[i,j])
    
        for i in range(12):
            if np.sum(AU_targets[i]) == 0 and np.sum(AU_preds[i]) == 0:
                F1s.append(0.0)
            else:
                F1s.append(f1_score(AU_targets[i], AU_preds[i]))

        F1s = np.array(F1s)
        F1_mean = np.mean(F1s)
        return F1_mean
    
    def metric_for_AU(self, y_pred, y_true):
        all_f1 = []
    
        for t in range(12):
            y_true_ = y_true[:, t]
            y_pred_ = y_pred[:, t]
            all_f1.append(f1_score(y_true_, y_pred_))
    
        f1_mean = np.mean(all_f1)
    
        AU_name_list = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
        all_f1 = dict(zip(AU_name_list, all_f1))
    
        return all_f1, f1_mean
    
    def au_f1(self, action_units_gt, action_units_pred):
        if isinstance(action_units_gt, torch.Tensor):
            action_units_gt = np.array(action_units_gt.detach().cpu())
        if isinstance(action_units_pred, torch.Tensor):
            action_units_pred = np.array(action_units_pred.detach().cpu())
            action_units_pred = (action_units_pred >= 0.5).astype(int)

        f1_scores = []

        for i in range(action_units_gt.shape[1]):
            f1 = f1_score(action_units_gt[:, i], action_units_pred[:, i], average='binary', zero_division=0)
            f1_scores.append(f1)

        mean_f1_score = np.mean(f1_scores)
        return mean_f1_score 
    
    def evaluate_overall(self, valence_true, valence_pred, arousal_true, arousal_pred, 
                         expression_true, expression_pred, action_units_true, action_units_pred):
        
        m_ccc = self.mean_ccc(valence_true, valence_pred, arousal_true, arousal_pred)
        expression_f1 = self.exp_f1(expression_true, expression_pred)
        action_units_f1 = f1_score(action_units_true, action_units_pred, labels=12, average='macro')

        P = m_ccc + expression_f1 + action_units_f1
        return P, m_ccc, expression_f1, action_units_f1

################################ Loss ######################################## 
class MTLLoss:
    def ccc_loss(self, y_true, y_pred, mask=None):
        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask

            if mask.sum() == 0:
                mask_sum = mask.sum() + 1e-8
                print("all data in the batch is invalid")
            else:
                mask_sum = mask.sum()

            y_true_mean = (y_true.sum() / mask_sum)
            y_pred_mean = (y_pred.sum() / mask_sum)

            y_true_var = ((y_true - y_true_mean) ** 2 * mask).sum() / mask_sum
            y_pred_var = ((y_pred - y_pred_mean) ** 2 * mask).sum() / mask_sum

            y_true_pred_cov = ((y_true - y_true_mean) * (y_pred - y_pred_mean) * mask).sum() / mask_sum
        else:
            y_true_mean = torch.mean(y_true)
            y_pred_mean = torch.mean(y_pred)
            y_true_var = torch.var(y_true)
            y_pred_var = torch.var(y_pred)
            y_true_pred_cov = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))

        ccc = (2 * y_true_pred_cov) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean)**2 + 1e-8)
        return 1 - ccc
    
    def va_ccc_loss(self, valence_pred, valence_true, arousal_pred, arousal_true, v_mask=None, a_mask=None):
        valence_loss = self.ccc_loss(valence_true, valence_pred, v_mask)
        arousal_loss = self.ccc_loss(arousal_true, arousal_pred, a_mask)
        total_loss = valence_loss + arousal_loss
        return total_loss
    
    def va_mse_loss(self, valence_pred, valence_true, arousal_pred, arousal_true):
        mse_loss = nn.MSELoss()
        valence_loss = mse_loss(valence_pred, valence_true)
        arousal_loss = mse_loss(arousal_pred, arousal_true)
        total_loss = valence_loss + arousal_loss
        return total_loss
    
    def expr_ce_loss(self, expr_pred, expr_gt, expr_masks=None):
        if expr_masks is not None:
            valid_indices = expr_masks.nonzero(as_tuple=True)[0]
            mask_stastic =  expr_masks.nonzero(as_tuple=True)[0].cpu().numpy().astype(int)
            if mask_stastic.sum() == 0:
                return torch.tensor(0.0).to(expr_pred.device)
                
            filtered_expr_pred = expr_pred[valid_indices]
            filtered_expr_gt = expr_gt[valid_indices]

            ce_loss = F.cross_entropy(filtered_expr_pred, filtered_expr_gt.long(), reduction='none')
            return ce_loss.mean()
        else:
            ce_loss = F.cross_entropy(expr_pred, expr_gt.long(), reduction='none')
            return ce_loss.mean()
    
    def au_bce_loss(self, au_pred, au_gt, au_masks=None):
        if au_masks is not None:
            valid_indices = au_masks.nonzero(as_tuple=True)[0]
            mask_stastic = au_masks.nonzero(as_tuple=True)[0].cpu().numpy().astype(int)
            if mask_stastic.sum() == 0:
                return torch.tensor(0.0).to(au_pred.device)
            
            filtered_expr_pred = au_pred[valid_indices]
            filtered_expr_gt = au_gt[valid_indices].float()

            bce_loss = F.binary_cross_entropy_with_logits(filtered_expr_pred, filtered_expr_gt, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(au_pred, au_gt.float(), reduction='none')
        
        return bce_loss.mean()

class BMCLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BMCLoss, self).__init__()
        self.margin = margin

    def forward(self, preds, targets):
        assert preds.shape == targets.shape # (batch_size, num_classes)
        batch_size, num_classes = preds.shape

        preds_expanded = preds.unsqueeze(1).expand(batch_size, batch_size, num_classes)
        targets_expanded = targets.unsqueeze(0).expand(batch_size, batch_size, num_classes)
        match_matrix = torch.sum(preds_expanded * targets_expanded, dim=2)

        positive_pairs = torch.diagonal(match_matrix)
        labels = torch.arange(batch_size).unsqueeze(1).to(preds.device)
        mask = torch.eq(labels, labels.T).float()

        positive_loss = F.relu(self.margin - positive_pairs.unsqueeze(1) + match_matrix) * mask
        negative_loss = F.relu(self.margin + positive_pairs.unsqueeze(1) - match_matrix) * (1 - mask)
        loss = positive_loss + negative_loss
        return loss
    
class CERLoss:
    def __init__(self, config):
        self.config = config

    def bce_loss(self, y_pred, y_gt):
        return F.binary_cross_entropy_with_logits(y_pred, y_gt)
    
    def bmc_loss(self, pred, target):
        """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
            pred: A float tensor of size [batch, d].
            target: A float tensor of size [batch, d].
            noise_var: A float number or tensor.
        Returns:
            loss: A float tensor. Balanced MSE Loss.
        """
        noise_var = self.config["noise_sigma"] ** 2

        # Ensure noise_var is a tensor
        if not isinstance(noise_var, torch.Tensor):
            noise_var = torch.tensor(noise_var, device=pred.device)

        I = torch.eye(pred.shape[-1], device=pred.device)
        logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.device))  # contrastive-like loss
        loss = loss * (2 * noise_var).detach()
        return loss

class CER_Evaluator:
    def __init__(self):
        self.label_map = {"Surprise": 1, "Fear": 2, "Disgust": 3, 
            "Happiness": 4, "Sadness": 5, "Anger": 6}

        self.composite_labels = [
            ("Fear", "Surprise"), ("Happiness", "Surprise"), 
            ("Sadness", "Surprise"), ("Disgust", "Surprise"), 
            ("Anger", "Surprise"), ("Sadness", "Fear"), 
            ("Sadness", "Anger")]
        
        
    def get_composite_probabilities(self, pred_probs):
        composite_probs = []
        for combo in self.composite_labels:
            combo_prob = pred_probs[:, self.label_map[combo[0]] - 1] + pred_probs[:, self.label_map[combo[1]] - 1]
            composite_probs.append(combo_prob.unsqueeze(1))
        return torch.cat(composite_probs, dim=1) # torch.Size([4, 7])
    
    def eval(self, pred, gt):
        pred_probs = torch.sigmoid(pred)
        composite_probs = self.get_composite_probabilities(pred_probs)
        pred_indices = composite_probs.argmax(dim=1) + 1  # torch.Size([48])
        f1 = f1_score(gt.cpu().numpy(), pred_indices.cpu().numpy(), average='macro')
        return f1
    
###############################   For Validation     #########################
    
import pandas as pd

def _read_from_csv(pred_txt):
    df = pd.read_csv(pred_txt, header=None)
    df_sorted = df.sort_values(by=0)
    img_names = df_sorted.iloc[:, 0:1].values
    valence = np.array(df_sorted.iloc[:, 1:2].values)
    arousal = np.array(df_sorted.iloc[:, 2:3].values)
    expression = np.array(df_sorted.iloc[:, 3:4].values)
    aus = np.array(df_sorted.iloc[:, 4:16].values)

    return img_names, arousal, valence, expression, aus

def _read_from_prob_csv(pred_txt):
    df = pd.read_csv(pred_txt, header=None)
    df_sorted = df.sort_values(by=0)
    img_names = df_sorted.iloc[:, 0:1].values
    valence = np.array(df_sorted.iloc[:, 1:2].values)
    arousal = np.array(df_sorted.iloc[:, 2:3].values)
    expression = np.array(df_sorted.iloc[:, 3:11].values)
    aus = np.array(df_sorted.iloc[:, 11:23].values)

    return img_names, arousal, valence, expression, aus
# (37381, 1) (37381, 1) (37381, 1) (37381, 8) (37381, 11)

def _read_from_AU_csv(pred_txt):
    df = pd.read_csv(pred_txt, header=None)
    preds = df.iloc[:, 1:13].values
    gts = df.iloc[:, 13:25].values
    preds_np = np.array(preds)
    gts_np = np.array(gts)
    print("len pred data:", len(preds_np))
    return preds_np, gts_np

def _read_from_EXPR_csv(pred_txt):
    df = pd.read_csv(pred_txt, header=None)
    preds = df.iloc[:, 1:2].values
    gts = df.iloc[:, 2:3].values
    preds_np = np.array(preds)
    gts_np = np.array(gts)
    print("len pred data:", len(preds_np))
    return preds_np, gts_np

def _read_from_VA_csv(pred_txt):
    df = pd.read_csv(pred_txt, header=None)
    v_pred = df.iloc[:, 1:2].values
    v_gt = df.iloc[:, 2:3].values
    a_pred = df.iloc[:, 3:4].values
    a_gt = df.iloc[:, 4:5].values

    v_pred_np = np.array(v_pred)
    v_gt_np = np.array(v_gt)
    a_pred_np = np.array(a_pred)
    a_gt_np = np.array(a_gt)

    print("len pred data:", len(a_pred_np))
    return v_pred_np, v_gt_np, a_pred_np, a_gt_np
    
def compute_AU_F1(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    AU_targets = [[] for i in range(12)]
    AU_preds = [[] for i in range(12)]
    F1s = []
    for i in range(pred.shape[0]):
        for j in range(12):
            p = pred[i,j]
            if p >= 0.5:
                AU_preds[j].append(1)
            else:
                AU_preds[j].append(0)
            AU_targets[j].append(label[i,j])
    
    for i in range(12):
        F1s.append(f1_score(AU_targets[i], AU_preds[i]))

    F1s = np.array(F1s)
    F1_mean = np.mean(F1s)
    return F1s, F1_mean

def au_f1(action_units_gt, action_units_pred, mask=None):
    
    if isinstance(action_units_gt, torch.Tensor):
        action_units_gt = action_units_gt.detach().cpu().numpy()
    if isinstance(action_units_pred, torch.Tensor):
        action_units_pred = action_units_pred.detach().cpu().numpy()

    action_units_pred = (action_units_pred >= 0.5).astype(int)

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        mask = mask[:, None]
        
        action_units_gt = action_units_gt * mask
        action_units_pred = action_units_pred * mask

    f1_scores = []

    for i in range(action_units_gt.shape[1]):
        if mask is not None:
            valid_indices = mask[:, 0] == 1  # 只需要检查第一个维度
        else:
            valid_indices = np.ones_like(action_units_gt[:, i], dtype=bool)
        
        if valid_indices.sum() > 0:
            f1 = f1_score(action_units_gt[valid_indices, i], action_units_pred[valid_indices, i], average='macro')
            f1_scores.append(f1)
        else:
            f1_scores.append(0.0)

    mean_f1_score = np.mean(f1_scores)
    return mean_f1_score
    
def final_evaluation_ori(pred_txt, config):
    mtl_func = MTLEvatuator()
    if config["AU"]:
        preds_np, gts_np = _read_from_AU_csv(pred_txt)
        F1s, F1_mean = compute_AU_F1_mask(preds_np, gts_np)
        return F1_mean
    
    if config["EXPR"]:
        preds_np, gts_np = _read_from_EXPR_csv(pred_txt)
        exp_f1score = mtl_func.exp_f1(gts_np, preds_np)
        return exp_f1score

    if config["VA"]:
        v_pred_np, v_gt_np, a_pred_np, a_gt_np = _read_from_VA_csv(pred_txt)
        mean_ccc_value, ccc_valence, ccc_arousal = mtl_func.mean_ccc(v_gt_np, v_pred_np, a_gt_np, a_pred_np)
        return mean_ccc_value, ccc_valence, ccc_arousal

def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        config = yaml.safe_load(file)
    return config

######################################## For Validation  Enhancement ####################################

from sklearn.metrics import f1_score
import numpy as np

def compute_AU_F1_mask(pred, label, mask=None):
    pred = np.array(pred)
    label = np.array(label)
    if mask is None:
        mask = np.full(pred.shape[0], True)
    mask = np.array(mask, dtype=bool)
    
    AU_targets = [[] for _ in range(12)]
    AU_preds = [[] for _ in range(12)]
    F1s = []
    
    for i in range(pred.shape[0]):
        if mask[i]:
            for j in range(12):
                p = pred[i, j]
                if p >= 0.5:
                    AU_preds[j].append(1)
                else:
                    AU_preds[j].append(0)
                AU_targets[j].append(label[i, j])
    
    for i in range(12):
        if AU_targets[i]:  
            F1s.append(f1_score(AU_targets[i], AU_preds[i]))
        else:
            F1s.append(0.0) 
    
    F1s = np.array(F1s)
    F1_mean = np.mean(F1s)
    return F1s, F1_mean


def final_evaluation(pred_txt, prob=False, config=None, gt_txt=None):
    mtl_func = MTLEvatuator()
    if gt_txt is None:
        gt_txt = config["val_anno_file"]
    if prob:
        img_names_pred, arousal_pred, valence_pred, expression_pred, aus_pred = _read_from_prob_csv(pred_txt)
    else:
        img_names_pred, arousal_pred, valence_pred, expression_pred, aus_pred = _read_from_csv(pred_txt)
        
    img_names_gt, arousal_gt, valence_gt, expression_gt, aus_gt = _read_from_csv(gt_txt)
    print("len img_names_pred", len(img_names_pred))
    print("len img_names_gt", len(img_names_gt))
    assert np.array_equal(img_names_pred, img_names_gt), "the sequence is not consistent!!!!"

    if config["AU"] or config is None:
        au_mask = np.all(aus_gt == -1, axis=-1).astype(float)
        au_mask = 1 - au_mask
        _, F1_mean = compute_AU_F1_mask(aus_pred, aus_gt, au_mask)
    else:
        F1_mean = 0
    
    if config["EXPR"] or config is None:
        expression_mask = (expression_gt != -1).astype(float)
        exp_f1score = mtl_func.exp_f1(expression_gt, expression_pred, expression_mask, prob=prob)
    else:
        exp_f1score = 0

    if config["VA_Arousal"] or config is None:
        arousal_mask = (arousal_gt != -5).astype(bool)
        ccc_arousal = mtl_func.ccc_cal(arousal_gt, arousal_pred, arousal_mask)
    else:
        ccc_arousal = 0
    
    if config["VA_Valence"] or config is None:
        valence_mask = (valence_gt != -5).astype(bool)
        ccc_valence = mtl_func.ccc_cal(valence_gt, valence_pred, valence_mask)
    else:
        ccc_valence = 0
    
    return F1_mean, exp_f1score, ccc_arousal, ccc_valence
    

    

