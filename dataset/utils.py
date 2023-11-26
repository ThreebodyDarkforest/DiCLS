# TODO: Add captions to labels
from .plant_pathology import PlantClassification, PlantKaggle
from torch.utils.data import DataLoader
from typing import Callable, Dict
import re
import numpy as np
from collections import defaultdict
from itertools import chain
import torch

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

def get_caption(tokenizer, idx2label, targets=None, 
                prefix=None, suffix=None, max_length=197):
    '''
    Convert (one hot) targets to tokenized captions, 
    meanwhile convert targets to trainable format
    example: [1 0 0 1 0 0] -> ([0 0 1 0 1], [4342, 32, 23, 43, 454]) # maybe \"Detect: cat, dog\"
    Note that the lenght of the returned targets are fixed to \"max_length\"
    '''
    caption = "All: " + ",".join([x for x in idx2label.values()]).replace("_", " ")

    if prefix:
        caption = prefix + '[SEP]' + caption
    if suffix:
        caption = caption + '[SEP]' + suffix
    
    tokens = tokenizer(
        caption,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    one_hot = None

    if targets is not None:
        bz = len(targets)
        tokens = tokenizer(
            [caption] * bz,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy().tolist()

        one_hot = np.zeros((len(targets), max_length))

        for id, target in enumerate(targets):
            labels = [idx2label[i].replace("_", " ") for i, c in enumerate(target) if c == 1]
            lab_tokens = tokenizer(labels,
                                   truncation=True,)['input_ids']
            lab_tokens = [token[1:-1] for token in lab_tokens]
            lab_tokens = list(chain(*lab_tokens))

            for j, token in enumerate(tokens['input_ids'].numpy()[0, :]):
                one_hot[id, j] = int(token in lab_tokens)
        
    return (tokens, one_hot)

def load_config(path):
    pass