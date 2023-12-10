from .plant_pathology import PlantClassification, PlantKaggle
from torch.utils.data import DataLoader
from typing import Callable, Dict

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