# TODO: Add captions to labels
from .plant_pathology import PlantClassification, PlantKaggle
from torch.utils.data import DataLoader
from typing import Callable, Dict
import re
from collections import defaultdict

all_datasets = {
    "PlantCLS": PlantClassification,
    "PlangKaggle": PlantKaggle,
}

def create_dataloader(name, data_path, batch_size: int = 32,
                      shuffle: bool = True, num_workers: int = 4,
                      collect_fn: Callable = None, **kwargs):
    dataset = all_datasets.get(name)(path=data_path, **kwargs)
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers, 
                      pin_memory=True, collate_fn=collect_fn)

def label2caption(label: str, infos: Dict):
    definition = infos.get('def')
    prefix = infos.get('prefix')
    desp = infos.get('desp')
    sep = infos.get('sep')

    caption = label.replace("_", " ")
    if prefix:
        caption = prefix[0] + prefix[1] + label
    if definition:
        caption = caption + sep + definition[0] + definition[1]
    if desp:
        caption = caption + sep + desp[0] + desp[1]
    return caption

def get_caption(ids, idx2label, infos):
    t = defaultdict(list)
    [t[id[0]].append(id[1]) for id in ids]
    captions = dict()
    for i, indexs in t.items():
        captions[i] = label2caption(','.join([idx2label[ind] 
                                    for ind in indexs]), infos)
    return captions

def load_config(path):
    pass