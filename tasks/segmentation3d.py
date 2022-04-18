import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class Segmentation3DTask(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.batch_size = self.hparams['batch_size']
        self.encoder = None


    def forward(self, x):
        # TODO
        raise NotImplementedError


    def training_step(self, batch, batch_nb):
        # TODO
        raise NotImplementedError


    def validation_step(self, batch, batch_nb):
        # TODO
        raise NotImplementedError


    def training_epoch_end(self, outputs):
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Train/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)


    def validation_epoch_end(self, outputs):
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Val/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)
            

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)


    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


    def configure_optimizers(self):
        lr = self.hparams['learning_rate']
        return [torch.optim.Adam(self.parameters(), lr=lr)]


    def train_dataloader(self):
        # TODO
        raise NotImplementedError
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
        
        return data_loader


    def val_dataloader(self):
        # TODO
        raise NotImplementedError
        data_loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=8)
        
        return data_loader


    def test_dataloader(self):
        # TODO
        raise NotImplementedError
        data_loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=8)
        
        return data_loader
