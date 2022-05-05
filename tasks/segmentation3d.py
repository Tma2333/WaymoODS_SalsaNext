import torch
import torch.nn as nn
from torchmetrics import JaccardIndex

from .base_task import BaseTask
from .utils import log_spherical_image_helper
from models import SphericalSegmentation

class Segmentation3DTask(BaseTask):
    def __init__ (self, params):
        # check for input data
        self.validate_input(params)
        super().__init__(params)

        self.spherical_model = SphericalSegmentation(self.hparams)
        
        self.jaccard = JaccardIndex(num_classes=self.hparams['num_cls'])

        self.criterion = nn.CrossEntropyLoss()

        self.sample_logged_epoch_train = -1
        self.sample_logged_epoch_val = -1

    
    def forward (self, x):
        return self.spherical_model(x)


    def training_step (self, batch, batch_nb):
        input = batch[self.input_key]
        label = batch[self.label_key][:, 1, :].long()

        if self.use_proj_pixel:
            label = self.mask_label_to_valid(input, label)

        out = self.forward(input)

        loss = self.compute_loss(out, label)
        
        pred = out.argmax(dim=1)
        mIoU = self.jaccard(pred, label)

        self.log_samples(input, label, pred, 'train')
        self.log("Train/R0_mIoU", mIoU, logger=True, prog_bar=True, rank_zero_only=True)
        self.log("Train/R0_loss", loss, logger=True, prog_bar=False, rank_zero_only=True)

        return {"loss" : loss, "mIoU" : mIoU}
        

    def validation_step (self, batch, batch_nb):
        input = batch[self.input_key]
        label = batch[self.label_key][:, 1, :].long()

        if self.use_proj_pixel:
            label = self.mask_label_to_valid(input, label)

        out = self.forward(input)

        loss = self.compute_loss(out, label)
        
        pred = out.argmax(dim=1)
        mIoU = self.jaccard(pred, label)

        self.log_samples(input, label, pred, 'val')

        return {"loss" : loss, "mIoU" : mIoU}


    def log_samples (self, input, label, pred, phase):
        if self.sample_logged_epoch_train < self.current_epoch and self.trainer.is_global_zero and phase=='train':
            if str(input.device) in ['cuda:0', 'cpu']: # check for dp
                if self.use_proj_pixel:
                    data = input[0].clone().movedim(0, -1)
                else: 
                    data = input[0][0]
                correct = label[0] == pred[0]
                fig = log_spherical_image_helper([data, label[0], pred[0], correct], vmin=[0,0,0,0], vmax=[1,22,22,1], 
                                                  cmap=['inferno', 'tab20', 'tab20', 'RdYlGn'])
                self.logger.experiment.add_figure('Train/sample', fig, self.current_epoch)
                self.sample_logged_epoch_train += 1
        if self.sample_logged_epoch_val < self.current_epoch and self.trainer.is_global_zero and phase=='val':
            if str(input.device) in ['cuda:0', 'cpu']: # check for dp
                if self.use_proj_pixel:
                    data = input[0].clone().movedim(0, -1)
                else: 
                    data = input[0][0]
                correct = label[0] == pred[0]
                fig = log_spherical_image_helper([data, label[0], pred[0], correct], vmin=[0,0,0,0], vmax=[1,22,22,1], 
                                                  cmap=['inferno', 'tab20', 'tab20', 'RdYlGn'])
                self.logger.experiment.add_figure('Val/sample', fig, self.current_epoch)
                self.sample_logged_epoch_val += 1


    def compute_loss (self, out, label):
        # out: [N, C, H, W]
        # label: [N, H, W]
        loss = self.criterion(out, label)
        return loss

    
    def mask_label_to_valid (self, input, label):
        # mask the label location where all channel of input are 0
        mask = torch.all(input == 0, dim=1)
        masked_label =  torch.masked_fill(label, mask, 0)
        return masked_label.type_as(label)


    def validate_input (self, params):
        if 'range_image' in params['key_to_load']:
            self.input_key = 'range_image'
            self.use_proj_pixel = False
        elif 'ri1_range_image' in params['key_to_load']:
            self.input_key = 'ri1_range_image'
            self.use_proj_pixel = False
        elif 'ri2_range_image' in params['key_to_load']:
            self.input_key = 'ri2_range_image'
            self.use_proj_pixel = False
        elif 'proj_pixel' in params['key_to_load']:
            self.input_key = 'proj_pixel'
            self.use_proj_pixel = True
        else:
            raise AttributeError(f'Not valid input in key_to_load. It mus include either range_image or proj_pixel')
        
        if 'ri1_label' in params['key_to_load']:
            self.label_key = 'ri1_label'
        elif 'ri2_label' in params['key_to_load']:
            self.label_key = 'ri2_label'
        else:
            raise AttributeError(f'This is supervised task, it must contain labels')
            