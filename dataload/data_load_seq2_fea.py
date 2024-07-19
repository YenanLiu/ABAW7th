import torch
import pandas as pd
import numpy as np
import os
import PIL
import yaml
import h5py

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        config = yaml.safe_load(file)
    return config

def _build_transforms(input_size, aug_ways, is_train=True):
    size = int(input_size)
    mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
    std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]
    if is_train:
        trans_list = [transforms.Resize(int(input_size), interpolation=Image.BICUBIC)]
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
    else:
        return transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

def _read_fea_h5(fea_path):
    features_dict = {}
    with h5py.File(fea_path, 'r') as h5f:
        for imgname in h5f.keys():
            img_group = h5f[imgname]
            for vname in img_group.keys():
                feature = img_group[vname][:]
                features_dict[(vname, imgname)] = feature
    return features_dict

class ABAWData(Dataset):
    def __init__(self, config, is_train=True):
        if is_train:
            self.anno_dir = config["train_anno_file"]
            self.data_dir = config["train_data_dir"]
        else:
            self.anno_dir = config["val_anno_file"]
            self.data_dir = config["val_data_dir"]

        self.data = pd.read_csv(self.anno_dir, header=None, dtype={1: 'float32'})
        self.sequence_len = config["seq_len"]
        self.window_len = config["win_len"]
        self.img_size = config["img_size"]

        self.video_indices = self.data.iloc[:, 0].str.split('/').str[0].unique()
        self.sequences_per_video = self._calculate_sequences_per_video()
        self.transform = _build_transforms(self.img_size, config["aug_ways"], is_train=is_train)

        if config["AU_Fea"]:
            fea_path = config["AU_Fea"] 
            self.au_fea = _read_fea_h5(fea_path)
        else:
            self.au_fea = None

        if config["EXPR_Fea"]:
            fea_path = config["EXPR_Fea"] 
            self.expr_fea = _read_fea_h5(fea_path)
        else:
            self.expr_fea = None
        
        if config["V_Fea"]:
            fea_path = config["V_Fea"] 
            self.v_fea = _read_fea_h5(fea_path)
        else:
            self.v_fea = None

    def _calculate_sequences_per_video(self):
        sequences_per_video = []
        for video_id in self.video_indices:
            video_data = self.data[self.data.iloc[:, 0].str.startswith(video_id)]
            num_frames = len(video_data)
            num_sequences = (num_frames + self.window_len - 1) // self.window_len
            sequences_per_video.append(num_sequences)
        return sequences_per_video

    def __len__(self):
        return sum(self.sequences_per_video)
    
    def __getitem__(self, idx):
        cumulative_sum = np.cumsum(self.sequences_per_video)
        video_idx = np.searchsorted(cumulative_sum, idx, side='right')
        sequence_idx = idx - (cumulative_sum[video_idx - 1] if video_idx > 0 else 0)

        video_id = self.video_indices[video_idx]
        video_data = self.data[self.data.iloc[:, 0].str.startswith(video_id)]
        frames = video_data.iloc[:, 1:].values
        frame_names = video_data.iloc[:, 0].values
        
        start = sequence_idx * self.window_len
        end = start + self.sequence_len
        sequence = frames[start:end]
        sequence_names = frame_names[start:end]
        
        if len(sequence) < self.sequence_len:
            last_frame = sequence[-1]
            padding = [last_frame] * (self.sequence_len - len(sequence))
            repeated_elements = np.full(self.sequence_len - len(sequence), sequence_names[-1], dtype=object)
            sequence = np.vstack([sequence] + padding)   
            sequence_names = np.concatenate((sequence_names, repeated_elements))
        
        img_tensor = [self.transform(Image.open(os.path.join(self.data_dir, name))) for name in sequence_names]
        # img_tensor = [self.transform(img) for img in images]

        # load fea
        if self.au_fea is not None: # features_dict[(vname, imgname)] = feature
            au_feas = [torch.tensor(self.au_fea[(name.split("/")[0], name.split("/")[-1].split(".")[0])]) for name in sequence_names]
            au_fea_tensor = torch.stack(au_feas)
        else:
            au_fea_tensor = None

        if self.expr_fea is not None: # features_dict[(vname, imgname)] = feature
            expr_feas = [torch.tensor(self.expr_fea[(name.split("/")[0], name.split("/")[-1].split(".")[0])]) for name in sequence_names]
            expr_fea_tensor = torch.stack(expr_feas)
        else:
            expr_fea_tensor = None
        
        if self.v_fea is not None: # features_dict[(vname, imgname)] = feature
            v_feas = [torch.tensor(self.v_fea[(name.split("/")[0], name.split("/")[-1].split(".")[0])]) for name in sequence_names]
            v_fea_tensor = torch.stack(v_feas)
        else:
            v_fea_tensor = None

        arousal = sequence[:, 0].astype(np.float32)
        valence = sequence[:, 1].astype(np.float32)
        expression = sequence[:, 2].astype(np.float32)
        au = sequence[:, 3:].astype(np.float32)

        return (torch.stack(img_tensor),
                torch.tensor(arousal, dtype=torch.float32),
                torch.tensor(valence, dtype=torch.float32),
                torch.tensor(expression, dtype=torch.long),
                torch.tensor(au, dtype=torch.float32),
                sequence_names, 
                au_fea_tensor,
                expr_fea_tensor,
                v_fea_tensor)

def collate_fn(batch):
    img_tensor = torch.stack([item[0] for item in batch]) 
    arousal = torch.stack([item[1] for item in batch])
    valence = torch.stack([item[2] for item in batch])
    expression = torch.stack([item[3] for item in batch])
    au = torch.stack([item[4] for item in batch])
    # frame_names = [frame_name for item in batch for frame_name in item[5]]
    frame_names = [item[5] for item in batch]

    if batch[0][6] is None:
        au_fea_tensor = None
    else:
        au_fea_tensor = torch.stack([item[6] for item in batch])

    if batch[0][7] is None:
        expr_fea_tensor = None
    else:
        expr_fea_tensor = torch.stack([item[7] for item in batch])

    if batch[0][8] is None:
        v_fea_tensor = None
    else:
        v_fea_tensor = torch.stack([item[8] for item in batch])

    return img_tensor, arousal, valence, expression, au, frame_names, au_fea_tensor, expr_fea_tensor, v_fea_tensor


class ABAWData_Eval(Dataset):
    def __init__(self, config):
        if config["TestMode"] == "val":
            self.anno_dir = config["val_anno_file"]
            self.data_dir = config["val_data_dir"]
        if config["TestMode"] == "test":
            self.anno_dir = config["test_anno_file"]
            self.data_dir = config["test_data_dir"]

        self.data = pd.read_csv(self.anno_dir, header=None, dtype={1: 'float32'})
        self.sequence_len = config["seq_len"]
        self.window_len = config["win_len"]
        self.img_size = config["img_size"]

        self.video_indices = self.data.iloc[:, 0].str.split('/').str[0].unique()
        self.sequences_per_video = self._calculate_sequences_per_video()
        self.transform = _build_transforms(self.img_size, config["aug_ways"], is_train=False)

        if config["AU_Fea"]:
            fea_path = config["AU_Fea"] 
            self.au_fea = _read_fea_h5(fea_path)
        else:
            self.au_fea = None

        if config["EXPR_Fea"]:
            fea_path = config["EXPR_Fea"] 
            self.expr_fea = _read_fea_h5(fea_path)
        else:
            self.expr_fea = None
        
        if config["V_Fea"]:
            fea_path = config["V_Fea"] 
            self.v_fea = _read_fea_h5(fea_path)
        else:
            self.v_fea = None

    def _calculate_sequences_per_video(self):
        sequences_per_video = []
        for video_id in self.video_indices:
            video_data = self.data[self.data.iloc[:, 0].str.startswith(video_id)]
            num_frames = len(video_data)
            num_sequences = (num_frames + self.window_len - 1) // self.window_len
            sequences_per_video.append(num_sequences)
        return sequences_per_video

    def __len__(self):
        return sum(self.sequences_per_video)
    
    def __getitem__(self, idx):
        cumulative_sum = np.cumsum(self.sequences_per_video)
        video_idx = np.searchsorted(cumulative_sum, idx, side='right')
        sequence_idx = idx - (cumulative_sum[video_idx - 1] if video_idx > 0 else 0)

        video_id = self.video_indices[video_idx]
        video_data = self.data[self.data.iloc[:, 0].str.startswith(video_id)]
        frames = video_data.iloc[:, 1:].values
        frame_names = video_data.iloc[:, 0].values
        
        start = sequence_idx * self.window_len
        end = start + self.sequence_len
        sequence = frames[start:end]
        sequence_names = frame_names[start:end]
        
        if len(sequence) < self.sequence_len:
            last_frame = sequence[-1]
            padding = [last_frame] * (self.sequence_len - len(sequence))
            repeated_elements = np.full(self.sequence_len - len(sequence), sequence_names[-1], dtype=object)
            sequence = np.vstack([sequence] + padding)   
            sequence_names = np.concatenate((sequence_names, repeated_elements))
        
        img_tensor = [self.transform(Image.open(os.path.join(self.data_dir, name))) for name in sequence_names]

        # load fea
        if self.au_fea is not None: # features_dict[(vname, imgname)] = feature
            au_feas = [torch.tensor(self.au_fea[(name.split("/")[0], name.split("/")[-1].split(".")[0])]) for name in sequence_names]
            au_fea_tensor = torch.stack(au_feas)
        else:
            au_fea_tensor = None

        if self.expr_fea is not None: # features_dict[(vname, imgname)] = feature
            expr_feas = [torch.tensor(self.expr_fea[(name.split("/")[0], name.split("/")[-1].split(".")[0])]) for name in sequence_names]
            expr_fea_tensor = torch.stack(expr_feas)
        else:
            expr_fea_tensor = None
        
        if self.v_fea is not None: # features_dict[(vname, imgname)] = feature
            v_feas = [torch.tensor(self.v_fea[(name.split("/")[0], name.split("/")[-1].split(".")[0])]) for name in sequence_names]
            v_fea_tensor = torch.stack(v_feas)
        else:
            v_fea_tensor = None

        return (torch.stack(img_tensor),
                sequence_names, 
                au_fea_tensor,
                expr_fea_tensor,
                v_fea_tensor)

def collate_fn_eval(batch):
    img_tensor = torch.stack([item[0] for item in batch]) 
    frame_names = [item[1] for item in batch]

    if batch[0][2] is None:
        au_fea_tensor = None
    else:
        au_fea_tensor = torch.stack([item[2] for item in batch])

    if batch[0][3] is None:
        expr_fea_tensor = None
    else:
        expr_fea_tensor = torch.stack([item[3] for item in batch])

    if batch[0][4] is None:
        v_fea_tensor = None
    else:
        v_fea_tensor = torch.stack([item[4] for item in batch])

    return img_tensor, frame_names, au_fea_tensor, expr_fea_tensor, v_fea_tensor


if __name__ == "__main__":
    # csv_file = '/root/code/ABAW/dataload/val.csv'

    config_file = "/root/code/ABAW/configs/Track1_enhance_VA_Fea.yml"
    config = read_yaml_to_dict(config_file)

    dataset = ABAWData( config, is_train=True)
    dataloader = DataLoader(dataset, batch_size=50, collate_fn=collate_fn, shuffle=False)

    names_set = set()
    for img_tensor, arousal, valence, expression, au, frame_names, au_fea_tensor, expr_fea_tensor, v_fea_tensor in dataloader:
        
        print("au_fea_tensor:", au_fea_tensor.shape)

        # print("Sequences: ", sequences) # torch.Size([4, 10, 3, 224, 224])
        # print("Arousal: ", arousal) torch.Size([4, 10])
        # print("Valence: ", valence) torch.Size([4, 10])
        # print("Expression: ", expression) torch.Size([4, 10])
        # print("AU: ", au.shape) #torch.Size([4, 10, 12])
        # if au.shape[-1] != 12:
        #     print(frame_names)
        # print("Frame Names: ", frame_names)
        # for frame_name in frame_names:
        #     for i in range(len(frame_name)):
        #         names_set.add(frame_name[i])
    # print("names_set", names_set)
    # print("len names_set", len(names_set))









