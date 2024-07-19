import random
import PIL
import torch
import os
import pandas as pd
import yaml

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        config = yaml.safe_load(file)
    return config

class ABAWDataset_Seq(Dataset):
    def __init__(self, config, data_file, root_dir, seq_len, input_size=112, is_train=True, AU_filter_Val=False, EXPR_filter_Val=False, VA_filter_Val=False):
        """
        data_file (str): the path of datafile
        root_dir (str): the path of data
        seq_len (int):
        """
        self.seq_len = seq_len
        self.is_train = is_train

        self.data = pd.read_csv(data_file, header=None) # header=None
        self.data.columns = ['image', 'valence', 'arousal', 'expression'] + [f'au{i}' for i in range(1, 13)]

        # filter out the data with invalid label
        if (config["AU"] and is_train) or AU_filter_Val: # 142382
            for i in range(1, 13):
                self.data = self.data[self.data[f'au{i}'] != -1] # 103316
        if (config["EXPR"] and is_train) or EXPR_filter_Val:
            self.data = self.data[self.data['expression'] != -1] # 52154
        if (config["VA"] and is_train) or VA_filter_Val: 
            self.data = self.data[(self.data['valence'] != -5) | (self.data['arousal'] != -5)]
        
        self.data.reset_index(drop=True, inplace=True)
        self.root_dir = root_dir
        self.transform = self._build_transforms(input_size)

        # sequence organize
        self.data['video_id'] = self.data['image'].apply(lambda x: x.split('/')[0])
        self.groups = {k: v for k, v in self.data.groupby('video_id').apply(lambda x: x.index.tolist()).items()}

        self.index_map = []
        for indices in self.groups.values():
            seq_num = len(indices) // self.seq_len
            remain_frames = len(indices) % self.seq_len
            for i in range(seq_num):
                self.index_map.append((indices[0] + i * (self.seq_len), indices[0] + (i+1) * (self.seq_len)))
            
            if remain_frames > 0 and (indices[-1] > indices[0] + seq_num * self.seq_len):
                self.index_map.append((indices[0] + seq_num * self.seq_len, indices[0] + seq_num * self.seq_len + remain_frames))
        
        if config["train_shuffle"]:
            random.shuffle(self.index_map)
        
    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self, idx):
        global_start, global_end = self.index_map[idx]
        selected_indices = list(range(global_start, global_end))
        if len(selected_indices) < self.seq_len and self.is_train and len(selected_indices) > 0:
            last_f = selected_indices[-1]
            selected_indices.extend([last_f] * (self.seq_len - len(selected_indices)))
            
        image_names = [self.data.loc[i, 'image'] for i in selected_indices]
        images = [Image.open(os.path.join(self.root_dir, self.data.loc[i, 'image'])) for i in selected_indices]
        images = [self.transform(img) for img in images]
        images = torch.stack(images)

        valence = torch.stack([torch.tensor(self.data.loc[i, 'valence'], dtype=torch.float) for i in selected_indices])
        arousal = torch.stack([torch.tensor(self.data.loc[i, 'arousal'], dtype=torch.float) for i in selected_indices])
        expression = torch.stack([torch.tensor(self.data.loc[i, 'expression'], dtype=torch.float) for i in selected_indices])
        aus = torch.stack([torch.tensor(self.data.loc[i, 'au1':'au12'].to_numpy().astype(float)) for i in selected_indices])
        return images, valence, arousal, expression, aus, image_names

    def _build_transforms(self, input_size):
        size = int(input_size)
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225] 

        return transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

if __name__ == "__main__":
    data_file = "/root/code/ABAW/dataload/train.csv"
    data_dir = "/root/code/ABAW/7thDataInfo/data_files/MTL/cropped_aligned"

    config_file = "/root/code/ABAW/configs/settings.yml"
    config = read_yaml_to_dict(config_file)

    dataset = ABAWDataset_Seq(config, data_file, data_dir, seq_len=50, input_size=224, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    df = pd.read_csv(data_file, header=None)
    name_from_csv = list(df[0])

    duplicates = list(set([x for x in name_from_csv if name_from_csv.count(x) > 1]))
    print(duplicates)
    num = 0
    read_name_set = set()
    for images, valence, arousal, expression, aus, image_names in dataloader:
        num += len(image_names)
        print(image_names, "\n")
        com_images = []
        for e in range(len(image_names[0])):
            for tup in image_names:
                com_images.add(tup[e])
        # for t in image_names:
        #     for e in t:
        #         read_name_set.add(e)
        # print("images.shape", images.shape, "\n")
        # print("valence.shape", valence.shape, "\n")
        # print("arousal.shape", arousal.shape, "\n")
        # print("expression.shape", expression.shape, "\n")
        # print("aus.shape", aus.shape, "\n")
                

    # difference = read_name_set.difference(name_from_csv)
    # print(difference)

    print("read file num:", num)
    # print("csv_file_num", len(name_from_csv))
 