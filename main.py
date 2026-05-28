import os
import torch
import utils.misc as misc
import pytorch_lightning as pl

from src import *
from argparse import ArgumentParser
from mmengine import Config, MODELS
from trainer.pl_model import pl_model
from trainer.data_dm import data_dm
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.load_pretrain import load_weights_by_keys

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_path', required=True, help="Configuration file path")
    parser.add_argument('--seed', type=int, default=7240, help='Random seed point')
    parser.add_argument('--log_interval', type=int, default=50, help='Log interval')
    parser.add_argument('--log_folder', type=str, default='debug', help='Log path')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for dataloader')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Check val every n epochs')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--align_method', type=str, default='scale', choices=['scale', 'metric'], help='Align method')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)
    cfg.update(vars(args))
    return cfg

if __name__ == '__main__':
    args = parse_args()
    
    """ Prepare training configs"""
    # set seed
    seed = args.seed
    pl.seed_everything(seed)
    
    # set log path
    log_folder = args.log_folder
    misc.check_path(log_folder)
    misc.check_path(os.path.join(log_folder, 'tensorboard'))
    if int(os.environ.get("RANK", "0")) == 0:
        args.dump(os.path.join(log_folder, 'config.py'))
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard')
    
    # set training logger
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=-1,
        filename='model-{epoch:02d}',
        save_last=True,
        save_weights_only=False
    )

    """ Prepare model and dataloader """
    model = MODELS.build(args.model)
    if args.resume_from is None:
        load_weights_by_keys(model, **args.weight_cfg, strict=False)
    elif int(os.environ.get("RANK", "0")) == 0:
        print(f"Resume from {args.resume_from}; skip weight_cfg loading.")

    model_wrapper = pl_model(
        optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler,
        model=model,
        loss_cfg=args.loss_cfg,
        align_method=args.align_method)
    
    data_module = data_dm(
        seed=args.seed,
        num_workers=args.num_workers,
        val_data=args.val_data,
        train_data=args.train_data,
        train_sampler_config=args.train_sampler_config)
    
    
    num_gpu = torch.cuda.device_count()
    num_nodes = args.num_nodes

    trainer = pl.Trainer(
        num_nodes=num_nodes,
        devices=[i for i in range(num_gpu)],
        strategy=DDPStrategy(
            accelerator='gpu',
            find_unused_parameters=True
        ),
        max_steps=args.training_steps,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step')
        ],
        precision="bf16-mixed",
        logger=tb_logger,
        sync_batchnorm=True,
        gradient_clip_val=1.0,
        log_every_n_steps=args.log_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch)
    
    # start training
    trainer.fit(model=model_wrapper, datamodule=data_module, ckpt_path=args.resume_from)