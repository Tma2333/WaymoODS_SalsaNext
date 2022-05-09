import yaml
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex

from .base_task import BaseTask
from .utils import log_spherical_image_helper
from models import SphericalSegmentation
from .loss import Lovasz_softmax, Poly1XentropyLoss

class Segmentation3DTask(BaseTask):
    def __init__ (self, params):
        # check for input data
        self.validate_input(params)
        super().__init__(params)

        self.spherical_model = SphericalSegmentation(self.hparams)
        
        self.jaccard = JaccardIndex(num_classes=self.hparams['num_cls'], ignore_index=0)

        self.sample_logged_epoch_train = -1
        self.sample_logged_epoch_val = -1

        # Loss initialization
        if self.hparams['weighted_Xentropy']:
            with open(str('./docs/cls_frequency.yaml'), 'r') as f:
                cls_freq = yaml.load(f, Loader = yaml.CLoader)
            cls_weight = torch.zeros(self.hparams['num_cls'])
            for cls_id, freq in cls_freq['cls_frequency'].items():
                cls_weight[cls_id] = freq
            cls_weight = 1 / (cls_weight + self.hparams['class_weight_eps'])
        else:
            cls_weight = None

        if self.hparams['lavasz_softmax']:
            self.lovasz = Lovasz_softmax(ignore=0)
        
        if self.hparams['poly1_Xentropy']:
            self.poly1Xentropy = Poly1XentropyLoss(eps=self.hparams['poly1_eps'], weight=cls_weight,
                                                   use_logit=False, ignore_index=0)
        else:
            # output is a probability
            self.NLL = torch.nn.NLLLoss(weight=cls_weight, ignore_index=0)

    
    def forward (self, x):
        return self.spherical_model(x)


    def training_step (self, batch, batch_nb):
        if self.fusion:
            input_list = []
            for in_key in self.input_key:
                input_list.append(batch[in_key])
            input = torch.concat(input_list, dim=1)
        else:
            input = batch[self.input_key]
        label = batch[self.label_key][:, 1, :].long()

        out = self.forward(input)

        loss = self.compute_loss(out, label)
        
        pred = out.argmax(dim=1)
        mIoU = self.jaccard(pred, label)

        self.log_samples(input, label, pred, 'train')
        self.log("Train/R0_mIoU", mIoU, logger=True, prog_bar=True, rank_zero_only=True)
        self.log("Train/R0_loss", loss, logger=True, prog_bar=False, rank_zero_only=True)

        return {"loss" : loss, "mIoU" : mIoU}
        

    def validation_step (self, batch, batch_nb):
        if self.fusion:
            input_list = []
            for in_key in self.input_key:
                input_list.append(batch[in_key])
            input = torch.concat(input_list, dim=1)
        else:
            input = batch[self.input_key]
        label = batch[self.label_key][:, 1, :].long()

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
                if self.fusion:
                    proj_im = input[0][-3:].clone().movedim(0, -1)
                correct = label[0] == pred[0]
                if self.fusion:
                    fig = log_spherical_image_helper([data, proj_im, label[0], pred[0], correct], vmin=[0,0,0,0,0], vmax=[1,1,22,22,1], 
                                                    cmap=['inferno', 'inferno', 'tab20', 'tab20', 'RdYlGn'])
                else:
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
                if self.fusion:
                    proj_im = input[0][-3:].clone().movedim(0, -1)
                correct = label[0] == pred[0]
                if self.fusion:
                    fig = log_spherical_image_helper([data, proj_im, label[0], pred[0], correct], vmin=[0,0,0,0,0], vmax=[1,1,22,22,1], 
                                                    cmap=['inferno', 'inferno', 'tab20', 'tab20', 'RdYlGn'])
                else:
                    fig = log_spherical_image_helper([data, label[0], pred[0], correct], vmin=[0,0,0,0], vmax=[1,22,22,1], 
                                                    cmap=['inferno', 'tab20', 'tab20', 'RdYlGn'])
                self.logger.experiment.add_figure('Val/sample', fig, self.current_epoch)
                self.sample_logged_epoch_val += 1
            

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, 
                             on_tpu=False, using_native_amp=False, using_lbfgs=False):
        peak_lr = self.hparams['peak_learning_rate']
        final_lr =  self.hparams['final_learning_rate']
        warmup_batch = self.hparams['lr_warmup_batch']
        decay_batch = self.hparams['lr_decay_batch']
        # warm up
        if self.trainer.global_step < warmup_batch:
            scale = min(1.0, float(self.trainer.global_step + 1) / warmup_batch)
            for pg in optimizer.param_groups:
                pg['lr'] = scale * peak_lr
        # decay
        elif warmup_batch <= self.trainer.global_step < decay_batch:
            scale = min(1.0, float(((self.trainer.global_step + 1) - warmup_batch) / (decay_batch-warmup_batch)))
            for pg in optimizer.param_groups:
                pg['lr'] = peak_lr + scale * (final_lr-peak_lr)

        optimizer.step(closure=optimizer_closure)


    def compute_loss (self, out, label):
        # out: [N, C, H, W]
        # label: [N, H, W]
        if self.hparams['lavasz_softmax']:
            lls = self.lovasz(out, label)
        else:
            lls = 0
        
        if self.hparams['poly1_Xentropy']:
            lce = self.poly1Xentropy(out, label)
        else:
            lce = self.NLL(torch.log(out.clamp(min=1e-8)), label)

        return lls + lce

    
    def mask_label_to_valid (self, input, label):
        # mask the label location where all channel of input are 0
        mask = torch.all(input == 0, dim=1)
        masked_label =  torch.masked_fill(label, mask, 0)
        return masked_label.type_as(label)


    def validate_input (self, params):
        self.input_key = None
        self.label_key = None
        self.fusion = False

        if 'range_image' in params['key_to_load']:
            self.input_key = 'range_image'
            self.use_proj_pixel = False
        elif 'ri1_range_image' in params['key_to_load']:
            self.input_key = 'ri1_range_image'
            self.use_proj_pixel = False
        elif 'ri2_range_image' in params['key_to_load']:
            self.input_key = 'ri2_range_image'
            self.use_proj_pixel = False
        
        if 'proj_pixel' in params['key_to_load']:
            if self.input_key is not None:
                self.input_key = [self.input_key]
                self.input_key.append('proj_pixel')
                self.fusion = True
            else:
                self.input_key = 'proj_pixel'
                self.use_proj_pixel = True
        
        if 'ri1_label' in params['key_to_load']:
            self.label_key = 'ri1_label'
        elif 'ri2_label' in params['key_to_load']:
            self.label_key = 'ri2_label'

        if self.input_key is None:
            raise AttributeError(f'Not valid input in key_to_load. It mus include either range_image or proj_pixel') 
        if self.label_key is None: 
            raise AttributeError(f'This is supervised task, it must contain labels')
            