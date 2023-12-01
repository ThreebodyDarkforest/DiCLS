from model.dicls import DiCLS
from model.utils.configs import Config
from model.backbone import GlobalEmbedding
from dataset.utils import label2caption, get_caption
from dataset.dataloader import create_dataloader
from dataset.transform import GeneralTransform
from model.utils.visualizer import vis_grid_attention
import torch
from argparse import ArgumentParser
from PIL import Image
import numpy as np

if __name__ == '__main__':
    train_dataloader = create_dataloader('PlantCLS', 'data/plant_dataset', 1, True, dataset_type='train', use_npz=True, label_smooth=0.01, mixup=0.1)
    for i, data in enumerate(train_dataloader):
        img, label, plain = data
        print(label, plain)