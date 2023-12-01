import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os.path as osp
import os
import pandas as pd
import numpy as np
from typing import List
from .transform import GeneralTransform
from . import utils
import random
import copy

class PlantDataset(Dataset):
    def __init__(self, img_path=None, label_path=None, npz_path=None,
                 dtype='train', one_hot=True, use_npz=False, label_smooth=None, mixup=None) -> None:
        super().__init__()

        assert dtype in ['train', 'test', 'val']
        assert (img_path and label_path) or npz_path, "Please select one way to load data."

        self.label_path, self.img_path = label_path, img_path
        self.mixup, self.label_smooth = mixup, label_smooth
        data = pd.read_csv(label_path)

        self.transform = GeneralTransform.train_transform \
                         if dtype == 'train' else GeneralTransform.test_transform
        self.use_npz = use_npz

        self.img_files, self.labels, all_labels = [], [], []
        self.plain_labels = []
        [all_labels.extend(label.split(' ')) for label in data['labels']]
        all_labels = np.unique(all_labels).tolist()
        self.idx2label = { k : v for k, v in enumerate(all_labels) }

        if not use_npz:
            for img, label in zip(data.get('image', data.get('images')), data['labels']):
                self.img_files.append(osp.join(self.img_path, img))
                if not one_hot:
                    self.labels.append([all_labels.index(x) for x in label.split(' ')])
                else:
                    self.labels.append(np.sum([np.eye(len(all_labels))[all_labels.index(x)] 
                                            for x in label.split(' ')], axis=0))
        else:
            data = np.load(npz_path)
            self.imgs, self.labels = data["imgs"], data["labels"]
            #self.idx2label = data["idx2label"]
            #if one_hot:
            #   self.labels = data["one_hot"]
        self.plain_labels = copy.deepcopy(self.labels)
        
        if one_hot and label_smooth is not None:
            self.labels = np.stack([utils.label_smooth(label, label_smooth) for label in self.labels])

        if one_hot and not use_npz:
            self.labels = np.stack(self.labels)

        self.n = len(self.labels)

    def _load(self, idx):
        if self.use_npz:
            img = self.imgs[idx]
        else:
            img = np.array(Image.open(self.img_files[idx]))
        
        label = self.labels[idx]
        return img, label
        
    def __getitem__(self, idx):
        img, label = self._load(idx)
        plain_label = self.plain_labels[idx]

        if self.mixup and random.random() < self.mixup:
            i = random.randint(0, self.n - 1)
            img, label = utils.mixup(img, label, *self._load(i))
            plain_label += self.plain_labels[i]
            plain_label[plain_label > 1] = 1

        img = self.transform(Image.fromarray(img))
            
        return img, label, plain_label
    
    def __len__(self):
        return len(self.imgs if self.use_npz else self.img_files)

class PlantKaggle(PlantDataset):
    def __init__(self, path, is_train=True, one_hot=True) -> None:
        self.img_path = osp.join(path, 'train_images') if is_train \
                        else osp.join(path, 'test_images')
        self.label_path = osp.join(path, 'train.csv')
        super().__init__(self.img_path, self.label_path)

        data = pd.read_csv(self.label_path)

        self.transform = GeneralTransform.train_transform \
                         if is_train else GeneralTransform.test_transform

        self.img_files, self.labels, all_labels = [], [], []
        [all_labels.extend(label.split(' ')) for label in data['labels']]
        all_labels = np.unique(all_labels).tolist()
        self.idx2label = { k : v for k, v in enumerate(all_labels) }

        for img, label in zip(data['image'], data['labels']):
            self.img_files.append(osp.join(self.img_path, img))
            if not one_hot:
                self.labels.append([all_labels.index(x) for x in label.split(' ')])
            else:
                self.labels.append(np.sum([np.eye(len(all_labels))[all_labels.index(x)] 
                                           for x in label.split(' ')], axis=0))
        if one_hot:
            self.labels = np.stack(self.labels)
        
    def __getitem__(self, index) -> List[np.ndarray]:
        img = Image.open(self.img_files[index])
        img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __repr__(self) -> str:
        return f"plant dataset locate at {self.img_path}."

class PlantClassification(PlantDataset):
    def __init__(self, path, dataset_type='train', use_npz=False, one_hot=True,
                 label_smooth=None, mixup=None) -> None:
        assert dataset_type in ['train', 'test', 'val']

        path = osp.join(path, dataset_type)
        img_path = osp.join(path, "images")
        label_path = osp.join(path, f"{dataset_type}_label.csv")
        npz_path = None

        if use_npz:
            npz_path = osp.join(path, f"{dataset_type}.npz")
        
        super().__init__(img_path, label_path, npz_path=npz_path,
                         use_npz=use_npz, one_hot=one_hot, dtype=dataset_type, 
                         label_smooth=label_smooth, mixup=mixup)

def collect_fn():
    pass

if __name__ == '__main__':
    test = PlantClassification('../data/plant_dataset/train')
    dataloader = DataLoader(test, 4, True, num_workers=4)
    for i in dataloader:
        print(i)