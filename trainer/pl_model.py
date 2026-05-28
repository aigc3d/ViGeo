from typing import Literal
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.loss.loss import loss_fn
from collections import defaultdict
from utils.metric_func import compute_depth_metrics, depth_meta_data

class pl_model(pl.LightningModule):
    def __init__(
        self,
        loss_cfg: dict,
        optimizer: dict,
        lr_scheduler: dict,
        model: nn.Module,
        align_method: Literal['scale', 'metric'] = 'scale',
    ):
        super().__init__()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.metrics = ['absrel', 'd1']
        self.loss_fn = loss_fn(**loss_cfg)
        self.align_method = align_method
        self.val_metrics = defaultdict(list)

    def _clear_cuda_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sync_train_epoch_state(self):
        train_dataloader = getattr(self.trainer, "train_dataloader", None)
        batch_sampler = getattr(train_dataloader, "batch_sampler", None)

        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(self.current_epoch)
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(self.current_epoch)

    def on_fit_start(self):
        self._clear_cuda_cache()

    def on_train_start(self):
        self._sync_train_epoch_state()
        self._clear_cuda_cache()

    def on_train_epoch_start(self):
        self._sync_train_epoch_state()
        self.val_metrics.clear()
        self._clear_cuda_cache()

    def training_step(self, sample, batch_idx):
        output = self.model(sample)

        loss_dict = self.loss_fn(output, sample)

        batch_size = sample['depth'].shape[0]

        for key in loss_dict.keys():
            self.log("iteration/" + key, loss_dict[key], batch_size=batch_size, on_epoch=True, sync_dist=True)
        total_loss = loss_dict['total']
        return total_loss
    
    def validation_step(self, sample, batch_idx):
        output = self.model(sample)
        data_name = sample['img_metas']['data_name'][0]
        depth_pred = output['depth_pred']
        depth_gt = sample['depth']
        metrics = compute_depth_metrics(depth_pred, depth_gt, metrics=self.metrics, align_method=self.align_method, **depth_meta_data[data_name])
        for key, value in metrics.items():
            self.val_metrics[key].append(value)
    
    def on_validation_epoch_end(self):
        for key, value in self.val_metrics.items():
            avg_value = sum(value) / len(value)
            self.log(
                "val/" + key,
                avg_value,
                on_epoch=True,
                sync_dist=True)
            value.clear()
        
    def configure_optimizers(self):
        if self.optimizer.type == 'AdamW':
            params_to_optimize = [
                {
                    "params": [param for name, param in self.model.named_parameters() if param.requires_grad and name.startswith("pretrained")],
                    "lr": self.optimizer.lr * 0.1,
                    "weight_decay": self.optimizer.weight_decay,
                    "name": "pretrained"
                },
                {
                    "params": [param for name, param in self.model.named_parameters() if param.requires_grad and not name.startswith("pretrained")],
                    "lr": self.optimizer.lr,
                    "weight_decay": self.optimizer.weight_decay,
                    "name": "others"}
                ]
        
            optimizer = torch.optim.AdamW(
                params_to_optimize
            )
        else:
            raise NotImplementedError
        
        if self.lr_scheduler.type == 'OneCycleLR':
            max_lr = [
                self.lr_scheduler.max_lr * 0.1,
                self.lr_scheduler.max_lr]
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=int(self.lr_scheduler.total_steps),
                pct_start=self.lr_scheduler.pct_start,
                cycle_momentum=self.lr_scheduler.cycle_momentum,
                anneal_strategy=self.lr_scheduler.anneal_strategy,
                final_div_factor=self.lr_scheduler.final_div_factor
            )
            interval=self.lr_scheduler.interval
            frequency=self.lr_scheduler.frequency
        else:
            raise NotImplementedError
        
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': interval,
            'frequency': frequency
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }