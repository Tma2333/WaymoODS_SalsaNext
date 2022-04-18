import os
from pathlib import Path

import yaml
import fire
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin, DataParallelPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from util import init_exp_folder
from task import get_task, oad_task,

def train(cfg):
    with open(str(cfg), 'r') as f:
        args = yaml.load(f, Loader = yaml.CLoader)
    task = get_task(args)
    
    # parse argument
    gpus = args['gpus']
    strategy = args['strategy']

    logger = TensorBoardLogger(args['save_dir'], name=args['exp_name'], version=args['version'])

    ckpt_dir = Path(args['save_dir']) / args['exp_name'] / f'version_{args['version']}' / 'ckpt'
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1, 
                              verbose=True, monitor=monitor, 
                              mode=args['monitor_mode'], every_n_epochs=1)

    earlystop_cb = EarlyStopping(monitor=args['monitor_metric'], 
                                 patience=args['patience'], 
                                 verbose=True, mode=args['monitor_mode'])
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if gpus == -1 or gpus > 1:
        if strategy == 'dp':
            strategy = DataParallelPlugin()
        if strategy == 'ddp':
            strategy = DDPPlugin(find_unused_parameters=False)
    else:
        strategy = None
    
    
    trainer = Trainer(gpus=gpus,
                      strategy=strategy,
                      logger=logger,
                      callbacks=[ckpt_cb, earlystop_cb, lr_monitor],
                      gradient_clip_val=args['gradient_clip_val'],
                      limit_train_batches=args['limit_train_batches'],
                      enable_model_summary = args['enable_model_summary'],
                      max_epochs=args['max_epochs'],
                      log_every_n_steps=1)
    trainer.fit(task)


if __name__ == "__main__":
    fire.Fire()
