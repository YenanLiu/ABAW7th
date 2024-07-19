import PIL
import torch
import os
import pandas as pd
import yaml
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        config = yaml.safe_load(file)
    return config

 
import pandas as pd
from torch.utils.data import Dataset

class ABAWDataset_Frame(Dataset):
    def __init__(self, config, is_train=True):
        """
        Initializes the dataset.
        
        Args:
            config (dict): Configuration dictionary with paths and parameters.
            is_train (bool): Flag indicating whether the dataset is for training or validation.
        """
        if is_train:
            self.anno_file = config["train_anno_file"]
            self.data_dir = config["train_data_dir"]
        else:
            self.anno_file = config["val_anno_file"]
            self.data_dir = config["val_data_dir"]

        # Read and process the annotation file
        self.data = pd.read_csv(self.anno_file, header=None)
        self.data.columns = ['image', 'valence', 'arousal', 'expression'] + [f'au{i}' for i in range(1, 13)]

        # Sort the data by the 'image' column
        self.data = self.data.sort_values(by='image')

        # Take every 5th row for training
        # if is_train:
        #     self.data = self.data.iloc[::5, :]
        #     self.data.reset_index(drop=True, inplace=True)

        self.is_train = is_train
        self.sequence_length = config["seq_len"]
        self.window_length = config["seq_len"]

        # Build image transformations
        self.transform = self._build_transforms(config["img_size"])

    def __len__(self):
        # Adjust the length to account for window length
        if len(self.data) < self.sequence_length:
            return 1
        return (len(self.data) - self.sequence_length) // self.window_length + 2
        # return len(self.data) // self.window_length + 1
    
    def __getitem__(self, idx):
        sequence_data = []
        start_idx = idx * self.window_length
        end_idx = start_idx + self.sequence_length

        # Ensure start_idx and end_idx are within bounds
        if end_idx > len(self.data):
            end_idx = len(self.data)
            start_idx = max(0, end_idx - self.sequence_length)

        for i in range(start_idx, end_idx):
            image_name = self.data.loc[i, 'image']
            image_path = os.path.join(self.data_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            valence = torch.tensor(self.data.iloc[i]['valence'], dtype=torch.float)
            arousal = torch.tensor(self.data.iloc[i]['arousal'], dtype=torch.float)
            expression = torch.tensor(self.data.iloc[i]['expression'], dtype=torch.float)
            aus_data = self.data.loc[i, 'au1':'au12'].to_numpy().astype(float)
            aus = torch.tensor(aus_data, dtype=torch.float)
            sequence_data.append((image, valence, arousal, expression, aus, image_name))

        # If the sequence length is less than required, repeat the last frame
        while len(sequence_data) < self.sequence_length and len(sequence_data) > 0:
            sequence_data.append(sequence_data[-1])

        images, valences, arousals, expressions, aus_list, image_names = zip(*sequence_data)
        
        return torch.stack(images), torch.stack(valences), torch.stack(arousals), torch.stack(expressions), torch.stack(aus_list), list(image_names)
    
    def _build_transforms(self, input_size):
        size = int(input_size)
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
        std = [0.229, 0.224, 0.225]  # ImageNet std

        return transforms.Compose([
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

# class ABAWDataset_Frame(Dataset):
#     def __init__(self, config, is_train=True):
#         """
#         Initializes the dataset.
        
#         Args:
#             config (dict): Configuration dictionary with paths and parameters.
#             is_train (bool): Flag indicating whether the dataset is for training or validation.
#         """
#         if is_train:
#             self.anno_file = config["train_anno_file"]
#             self.data_dir = config["train_data_dir"]
#         else:
#             self.anno_file = config["val_anno_file"]
#             self.data_dir = config["val_data_dir"]

#         # Read and process the annotation file
#         self.data = pd.read_csv(self.anno_file, header=None)
#         self.data.columns = ['image', 'valence', 'arousal', 'expression'] + [f'au{i}' for i in range(1, 13)]

#         # Sort the data by the 'image' column
#         self.data = self.data.sort_values(by='image')

#         # Take every 5th row
#         if is_train:
#             self.data = self.data.iloc[::10, :]
#             self.data.reset_index(drop=True, inplace=True)

#         self.is_train = is_train

#         # Build image transformations
#         self.transform = self._build_transforms(config["img_size"])

#     def __len__(self):
#         return len(self.data) 
    
#     def __getitem__(self, i):
#         image_name = self.data.loc[i, 'image']
#         image = Image.open(os.path.join(self.data_dir , self.data.loc[i, 'image'])) 
#         image = self.transform(image)
 
#         valence = torch.tensor(self.data.iloc[i]['valence'], dtype=torch.float)
#         arousal = torch.tensor(self.data.iloc[i]['arousal'].tolist(), dtype=torch.float)
#         expression = torch.tensor(self.data.iloc[i]['expression'].tolist(), dtype=torch.float)
#         aus_data = self.data.loc[i, 'au1':'au12'].to_numpy()
#         arr_float = aus_data.astype(float)
#         aus = torch.tensor(arr_float, dtype=torch.float)

#         return image, valence, arousal, expression, aus, image_name
    
#     def _build_transforms(self, input_size):
#         size = int(input_size)
#         mean = [0.485, 0.456, 0.406]  # ImageNet mean
#         std = [0.229, 0.224, 0.225]  # ImageNet std

#         return transforms.Compose([
#             transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
#             # transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

################################# Track 2 ###########################################

def to_one_hot(label_index_list, num_classes):
    one_hot = [0] * num_classes
    for index in label_index_list:
        if index > 0:
            one_hot[index - 1] = 1
    return one_hot

def _build_transforms(input_size):
    size = int(input_size)
    mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
    std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]

    return transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

label2ID = {"Surprise":1, "Fear":2, "Disgust":3, "Happiness":4, "Sadness":5, "Anger":6}
com_multi2ID = {1: [2, 1], 2: [4, 1], 3: [5, 1], 4: [3, 1], 5: [6, 1], 6: [5, 2], 7: [5, 6]}
# "Fear Surprise" "Happiness Surprise" "Sadness Surprise" "Disgust Surprise"  "Anger Surprise" "Sadness Fear" "Sadness Anger"

def update_image_path(image_name, root_dir):
    new_path = os.path.join(root_dir, image_name)
    return new_path

class ABAW_Track2_single_Label_Frame(Dataset):
    def __init__(self, config, is_train):
        super().__init__()
        self.rab_flag = config["Data_RAF-DB-Single"]
        self.affwild_flag = config["Data_Aff-wild2"]
        self.is_train = is_train

        rab_single_datadir = config["rab_single_datadir"]
        rab_single_file = config["rab_single_train_file"]
        rab_single_data = pd.read_csv(rab_single_file, header=None)
        rab_single_data[0] = rab_single_data[0].apply(update_image_path, root_dir=rab_single_datadir)

        affwild_datadir = config["affwild_datadir"]
        affwild_file = config["affwild_train_file"]
        affwild_data = pd.read_csv(affwild_file, header=None)
        affwild_data[0] = affwild_data[0].apply(update_image_path, root_dir=affwild_datadir)

        competition_datadir = config["competition_datadir"]
        competition_val_file = config["competition_val_file"]

        if not is_train:
            self.data = pd.read_csv(competition_val_file, header=None)
            self.data[0] = self.data[0].apply(update_image_path, root_dir=competition_datadir)
        else:
            pd_frame_list = []
            if self.rab_flag:
                pd_frame_list.append(rab_single_data)
            if self.affwild_flag:
                pd_frame_list.append(affwild_data)
            if len(pd_frame_list) > 0:
                self.data = pd.concat(pd_frame_list)

        self.data.columns = ['image', 'label']

        if config["aug_classical"]:
            self.transform = transform_enhance(config["aug_ways"], config["input_size"])
        else:
            self.transform = _build_transforms(config["input_size"])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        image_path = self.data.iloc[i, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
    
        if self.is_train:
            labels = self.data.iloc[i, 1].split(" ")
            label_index_list = []
            for label in labels:
                if label == '':
                    continue
                label_index_list.append(label2ID[label])
            label_gt = torch.tensor(to_one_hot(label_index_list, 6), dtype=torch.float)
        else:
            labels = self.data.iloc[i, 1]
            label_gt = torch.tensor(labels, dtype=torch.float)

        return image, label_gt

class ABAW_Track2_Multi_Label_Frame(Dataset):
    def __init__(self, config):
        super().__init__()
        # self.data = pd.read_csv(data_file, header=None)
        self.rab_multi_flag = config["Data_RAF-DB-Multi"]
        self.competition = config["Data_Competition"]

        data_frame_list = []
        if self.rab_multi_flag:
            rab_compound_datadir = config["rab_compound_datadir"]
            rab_compound_file = config["rab_compound_train_file"]
            rab_compound_train_data = pd.read_csv(rab_compound_file, header=None)
            rab_compound_train_data[0] = rab_compound_train_data[0].apply(update_image_path, root_dir=rab_compound_datadir)

            data_frame_list.append(rab_compound_train_data)

        if self.competition:
            competition_datadir = config["competition_datadir"]
            competition_train_file = config["competition_train_file"]
            competition_train_data = pd.read_csv(competition_train_file, header=None)
            competition_train_data[0] = competition_train_data[0].apply(update_image_path, root_dir=competition_datadir)

            data_frame_list.append(competition_train_data)

        self.data = pd.concat(data_frame_list)
        self.data.columns = ['image', 'label']

        if config["aug_classical"]:
            self.transform = transform_enhance(config["aug_ways"], config["input_size"])
        else:
            self.transform = _build_transforms(config["input_size"])
    
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, i):
        image_path = self.data.iloc[i, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if isinstance(self.data.iloc[i, 1], np.int64) or isinstance(self.data.iloc[i, 1], int): 
            labels = self.data.iloc[i, 1]
            label_index_list = com_multi2ID[labels]
        else:
            labels = self.data.iloc[i, 1].split(" ")
            label_index_list = [label2ID[label] for label in labels]

        label_one_hot = torch.tensor(to_one_hot(label_index_list, 6), dtype=torch.float)

        return image, label_one_hot

if __name__ == "__main__":
    # data_file = "/root/code/ABAW/dataload/train.csv"
    # data_dir = "/root/code/ABAW/7thDataInfo/data_files/MTL/cropped_aligned"

    config_file = "/root/code/ABAW/configs/Track1_enhance_VA.yml"
    config = read_yaml_to_dict(config_file)
    dataset = ABAWDataset_Frame(config, is_train=True)
    dataloader = DataLoader(dataset, batch_size=config["batchsize"],  shuffle=False, num_workers=4)
    print("len dataloader:", len(dataloader))
    names_set = set()
    total_sum = 0
    for img_tensor, arousal, valence, expression, au, frame_names in dataloader:
        print("AU: ", au.shape)
        new_images_name = [name for img_name in frame_names for name in img_name ]
        names_set.update(set(new_images_name))
        total_sum += (img_tensor.shape[0] * img_tensor.shape[1])
        print(total_sum)
    print("name number:", len(names_set))
    # dataset = ABAW_Track2_single_Label_Frame(config, is_train=True)
    # # dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    # # dataset = ABAW_Track2_Multi_Label_Frame(config)
    # dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
 
    # for image, label in dataloader:

    #     print("*************************************************************")

 