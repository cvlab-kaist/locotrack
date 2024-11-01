import os
import configparser
import argparse
import logging
from functools import partial
from typing import Any, Dict, Optional, Union, Tuple

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import torch
from torch.utils.data import DataLoader

from data.kubric_data import KubricData
from data.real_data import RealVideoDataset
from models.locotrack_model import LocoTrack
import model_utils
from data.evaluation_datasets import get_eval_dataset


class LocoTrackModel(L.LightningModule):
    def __init__(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_forward_kwargs: Optional[Dict[str, Any]] = None,
        loss_name: Optional[str] = 'tapir_loss',
        loss_kwargs: Optional[Dict[str, Any]] = None,
        query_first: Optional[bool] = False,
        optimizer_name: Optional[str] = 'Adam',
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_name: Optional[str] = 'OneCycleLR',
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = LocoTrack(**(model_kwargs or {}))
        self.model_forward_kwargs = model_forward_kwargs or {}
        self.loss = partial(model_utils.__dict__[loss_name], **(loss_kwargs or {}))
        self.query_first = query_first

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 2e-3}
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs or {'max_lr': 2e-3, 'pct_start': 0.05, 'total_steps': 300000}

    def training_step(self, batch, batch_idx):
        output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        loss, loss_scalars = self.loss(batch, output)
        
        self.log_dict(
            {f'train/{k}': v.item() for k, v in loss_scalars.items()},
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        opt = self.optimizers()
        sched = self.lr_schedulers()
        opt.zero_grad()
        self.manual_backward(loss)

        if any([p.grad.isnan().any() for p in self.parameters() if p.grad is not None]):
            print('nan gradients detected, skipping step')
            [p.grad.zero_() for p in self.parameters() if p.grad is not None and p.grad.isnan().any()]

        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

        opt.step()
        sched.step()

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        loss, loss_scalars = self.loss(batch, output)
        metrics = model_utils.eval_batch(batch, output, query_first=self.query_first)
        
        log_prefix = 'val/'
        if dataloader_idx is not None:
            log_prefix = f'val/data_{dataloader_idx}/'

        self.log_dict(
            {log_prefix + k: v for k, v in loss_scalars.items()},
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            {log_prefix + k: v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
        )
        logging.info(f"Batch {batch_idx}: {metrics}")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        loss, loss_scalars = self.loss(batch, output)
        metrics = model_utils.eval_batch(batch, output, query_first=self.query_first)

        log_prefix = 'test/'
        if dataloader_idx is not None:
            log_prefix = f'test/data_{dataloader_idx}/'
        
        self.log_dict(
            {log_prefix + k: v for k, v in loss_scalars.items()},
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            {log_prefix + k: v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
        )
        logging.info(f"Batch {batch_idx}: {metrics}")
        
    def configure_optimizers(self):
        trainable_params = []

        for k, v in self.named_parameters():
            if 'dino' not in k or 'lora' in k:
                trainable_params.append(v)
            else:
                v.requires_grad = False

        optimizer = torch.optim.__dict__[self.optimizer_name](
            [
                {'params': trainable_params, **self.optimizer_kwargs},
            ]
        )
        scheduler = torch.optim.lr_scheduler.__dict__[self.scheduler_name](optimizer, **self.scheduler_kwargs)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def train(
    mode: str,
    save_path: str,
    val_dataset_path: str,
    ckpt_path: str = None,
    kubric_dir: str = '',
    precision: str = '32',
    batch_size: int = 1,
    val_check_interval: Union[int, float] = 5000,
    log_every_n_steps: int = 10,
    gradient_clip_val: float = 1.0,
    max_steps: int = 300_000,
    model_kwargs: Optional[Dict[str, Any]] = None,
    model_forward_kwargs: Optional[Dict[str, Any]] = None,
    loss_name: str = 'tapir_loss',
    loss_kwargs: Optional[Dict[str, Any]] = None,
    optimizer_name: str = 'Adam',
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    scheduler_name: str = 'OneCycleLR',
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    real_data_dir: str = '',
    real_video_size: Optional[Tuple[int, int]] = (512, 512),
    real_video_length: int = 24,
    real_points_to_sample: int = 512,
    real_batch_size: int = 1,
    # query_first: bool = False,
):
    """Train the LocoTrack model with specified configurations."""
    seed_everything(42, workers=True)

    model = LocoTrackModel(
        model_kwargs=model_kwargs,
        model_forward_kwargs=model_forward_kwargs,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
        query_first='q_first' in mode,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
    )
    if ckpt_path is not None and 'train' in mode:
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    logger = WandbLogger(project='LocoTrack_Pytorch', save_dir=save_path, id=os.path.basename(save_path))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_last=True,
        save_top_k=3,
        mode="max",
        monitor="val/average_pts_within_thresh",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
    )

    eval_dataset = get_eval_dataset(
        mode=mode,
        path=val_dataset_path,
    )
    eval_dataloder = {
        k: DataLoader(
            v,
            batch_size=1,
            shuffle=False,
        ) for k, v in eval_dataset.items()
    }

    if 'train' in mode:
        trainer = L.Trainer(
            strategy='ddp',
            logger=logger,
            precision=precision,
            val_check_interval=val_check_interval,
            log_every_n_steps=log_every_n_steps,
            max_steps=max_steps,
            sync_batchnorm=True,
            callbacks=[checkpoint_callback, lr_monitor],
        )
        synthetic_dataloader = KubricData(
            global_rank=trainer.global_rank, 
            world_size=trainer.world_size,
            data_dir=kubric_dir, 
            batch_size=batch_size * trainer.world_size,
        )
        real_dataset = RealVideoDataset(
            data_dir=real_data_dir,
            video_size=real_video_size,
            video_length=real_video_length,
            points_to_sample=real_points_to_sample,
        )
        real_dataloader = DataLoader(
            real_dataset, 
            batch_size=real_batch_size,
            shuffle=True,
            num_workers=16,
        )
        dataloader = CombinedLoader({
            'synthetic': synthetic_dataloader,
            'real': real_dataloader,
        }, 'max_size_cycle')
        trainer.fit(model, dataloader, eval_dataloder, ckpt_path=ckpt_path)
    elif 'eval' in mode:
        trainer = L.Trainer(strategy='ddp', logger=logger, precision=precision)
        trainer.test(model, eval_dataloder, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Invalid mode: {mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate the LocoTrack model.")
    parser.add_argument('--config', type=str, default='config.ini', help="Path to the configuration file.")
    parser.add_argument('--mode', type=str, required=True, help="Mode to run: 'train' or 'eval' with optional 'q_first' and the name of evaluation dataset.")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument('--save_path', type=str, default='snapshots', help="Path to save the logs and checkpoints.")
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Extract parameters from the config file
    train_params = {
        'mode': args.mode,
        'ckpt_path': args.ckpt_path,
        'save_path': args.save_path,
        'val_dataset_path': eval(config.get('TRAINING', 'val_dataset_path', fallback='{}')),
        'kubric_dir': config.get('TRAINING', 'kubric_dir', fallback=''),
        'precision': config.get('TRAINING', 'precision', fallback='32'),
        'batch_size': config.getint('TRAINING', 'batch_size', fallback=1),
        'val_check_interval': config.getfloat('TRAINING', 'val_check_interval', fallback=5000),
        'log_every_n_steps': config.getint('TRAINING', 'log_every_n_steps', fallback=10),
        'gradient_clip_val': config.getfloat('TRAINING', 'gradient_clip_val', fallback=1.0),
        'max_steps': config.getint('TRAINING', 'max_steps', fallback=300000),
        'model_kwargs': eval(config.get('MODEL', 'model_kwargs', fallback='{}')),
        'model_forward_kwargs': eval(config.get('MODEL', 'model_forward_kwargs', fallback='{}')),
        'loss_name': config.get('LOSS', 'loss_name', fallback='tapir_loss'),
        'loss_kwargs': eval(config.get('LOSS', 'loss_kwargs', fallback='{}')),
        'optimizer_name': config.get('OPTIMIZER', 'optimizer_name', fallback='Adam'),
        'optimizer_kwargs': eval(config.get('OPTIMIZER', 'optimizer_kwargs', fallback='{"lr": 2e-3}')),
        'scheduler_name': config.get('SCHEDULER', 'scheduler_name', fallback='OneCycleLR'),
        'scheduler_kwargs': eval(config.get('SCHEDULER', 'scheduler_kwargs', fallback='{"max_lr": 2e-3, "pct_start": 0.05, "total_steps": 300000}')),
        'real_data_dir': config.get('REALDATA', 'data_dir', fallback=''),
        'real_video_size': eval(config.get('REALDATA', 'video_size', fallback='(512, 512)')),
        'real_video_length': config.getint('REALDATA', 'video_length', fallback=24),
        'real_points_to_sample': config.getint('REALDATA', 'points_to_sample', fallback=512),
        'real_batch_size': config.getint('REALDATA', 'batch_size', fallback=1),
    }

    train(**train_params)
