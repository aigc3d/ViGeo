import torch
import pytorch_lightning as pl
from mmengine import DATASETS
from .sampler import MultiScaleBatchSampler
from torch.utils.data.dataloader import DataLoader

class data_dm(pl.LightningDataModule):
    def __init__(
        self,
        seed: int,
        num_workers: int,
        val_data: dict,
        train_data: dict,
        train_sampler_config: dict
    ):
        super().__init__()
        self.train_sampler_config = train_sampler_config
        self.train_dataset = DATASETS.build(train_data)
        self.val_dataset = DATASETS.build(val_data)

        self.seed = seed
        self.num_workers = num_workers
    
    def train_dataloader(self):
        sampler = MultiScaleBatchSampler(
            dataset_lengths=[len(d) for d in self.train_dataset.datasets],
            seed=self.seed,
            **self.train_sampler_config)

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False)
        
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            drop_last=False)
        
        return val_dataloader