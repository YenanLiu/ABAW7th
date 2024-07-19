import numpy as np
import torch

from torchvision import transforms
from PIL import Image

class Mixup:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.target_combinations = {(6, 3), (6, 1), (3, 1), (2, 6), (2, 1), (4, 3), (4, 1), (5, 6), (5, 3), (5, 2), (5, 1)}

    def __call__(self, x, y):

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        y_a, y_b = y, y[index]
        
        # Check if the label combinations are in the target list
        while not self.is_valid_combination(y_a, y_b):
            index = torch.randperm(batch_size)
            y_b = y[index]
        
        mixed_x = self.alpha * x + (1 - self.alpha) * x[index, :]
        
        return mixed_x, y_a, y_b, self.alpha

    def is_valid_combination(self, y_a, y_b):
        # Check if all label combinations are in the target list
        for a, b in zip(y_a, y_b):
            if (a.item(), b.item()) not in self.target_combinations and (b.item(), a.item()) not in self.target_combinations:
                return False
        return True

    def mixup_criterion(self, criterion, pred, y_a, y_b):
        return self.alpha * criterion(pred, y_a) + (1 - self.alpha) * criterion(pred, y_b)
 

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class CutMix(object):
    def __init__(self):
        self.target_combinations = {(6, 3), (6, 1), (3, 1), (2, 6), (2, 1), (4, 3), (4, 1), (5, 6), (5, 3), (5, 2), (5, 1)}

    def __call__(self, x, y): # x: imgs, y: labels
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        y_a, y_b = y, y[index]

        while not self.is_valid_combination(y_a, y_b):
            index = torch.randperm(batch_size)
            y_b = y[index]

        # 拼接x 和 x[index, :]
        mixed_x = x.clone()
        _, _, _, W = x.size()
        mixed_x[:, :, :, :W//2] = x[:, :, :, :W//2]  # 左半边
        mixed_x[:, :, :, W//2:] = x[index, :, :, W//2:]  # 右半边

        return mixed_x, y_a, y_b
    
    def is_valid_combination(self, y_a, y_b):
        for a, b in zip(y_a, y_b):
            if (a.item(), b.item()) not in self.target_combinations and (b.item(), a.item()) not in self.target_combinations:
                return False
        return True

def transform_enhance(aug_ways, input_size):
    mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
    std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]

    if "RandomHorizontal" in aug_ways:
        trans_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if "RandomResizedCrop" in aug_ways:
        trans_list.append(transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)))
    if "RandomRotation" in aug_ways:
        trans_list.append(transforms.RandomRotation(degrees=10))
    if "ColorJitter" in aug_ways:
        trans_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    if "RandomAffine" in aug_ways:
        trans_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))

    trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(trans_list)
    
    