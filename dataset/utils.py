# TODO: Add captions to labels
from typing import Callable, Dict
import re
import numpy as np
from collections import defaultdict
from itertools import chain
import torch
import cv2

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
    caption = "Detect: " + ". ".join([x for x in idx2label.values()]).replace("_", " ")

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

def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    #labels = np.concatenate((labels, labels2), 0)
    labels = (labels * r + labels2 * (1 - r))
    return im, labels

def label_smooth(labels, label_smooth=0.02, num_classes=None):
    if num_classes is None:
        num_classes = labels.shape[-1]
    
    return (1 - label_smooth) * labels + label_smooth / num_classes