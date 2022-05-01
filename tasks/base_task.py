import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data import H5Dataset


class BaseTask(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.batch_size = self.hparams['batch_size']


    def forward(self, x):
        # TODO
        raise NotImplementedError


    def training_step(self, batch, batch_nb):
        # TODO
        raise NotImplementedError


    def validation_step(self, batch, batch_nb):
        # TODO
        raise NotImplementedError


    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)


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


    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


    def configure_optimizers(self):
        lr = self.hparams['learning_rate']
        return [torch.optim.Adam(self.parameters(), lr=lr)]


    def train_dataloader(self):
        h5_path = self.hparams['train_path']
        key_to_load = self.hparams['key_to_load']
        dataset = H5Dataset(h5_path, key_to_load, test=False)
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.hparams['loader_worker'])
        
        return data_loader


    def val_dataloader(self):
        h5_path = self.hparams['val_path']
        key_to_load = self.hparams['key_to_load']
        dataset = H5Dataset(h5_path, key_to_load, test=False)
        data_loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.hparams['loader_worker'])
        
        return data_loader


    def test_dataloader(self):
        h5_path = self.hparams['test_path']
        key_to_load = self.hparams['key_to_load']
        dataset = H5Dataset(h5_path, key_to_load, test=True)
        data_loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.hparams['loader_worker'])
        
        return data_loader
