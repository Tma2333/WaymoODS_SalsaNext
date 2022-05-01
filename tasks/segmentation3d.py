import torch
import torch.nn as nn
from torchmetrics import JaccardIndex

from .base_task import BaseTask
from models import SphericalSegmentation

class Segmentation3DTask(BaseTask):
    def __init__(self, params):
        super().__init__(params)

        self.spherical_model = SphericalSegmentation(self.hparams)
        
        self.jaccard = JaccardIndex(num_classes=self.hparams['num_cls'])

        self.criterion = nn.CrossEntropyLoss()

    
    def forward(self, x):
        return self.spherical_model(x)


    def training_step(self, batch, batch_nb):
        input = batch['range_image']
        label = batch['ri1_label'][:, 1, :].long()

        out = self.forward(input)

        loss = self.compute_loss(out, label)
        
        pred = out.argmax(dim=1)
        mIoU = self.jaccard(pred, label)

        self.log("Train/R0_mIoU", mIoU, logger=True, prog_bar=True, rank_zero_only=True)
        self.log("Train/R0_loss", loss, logger=True, prog_bar=False, rank_zero_only=True)

        return {"loss" : loss, "mIoU" : mIoU}


    def compute_loss (self, out, label):
        # out: [N, C, H, W]
        # label: [N, H, W]
        loss = self.criterion(out, label)
        return loss
        

    def validation_step(self, batch, batch_nb):
        input = batch['range_image']
        label = batch['ri1_label'][:, 1, :].long()

        out = self.forward(input)

        loss = self.compute_loss(out, label)
        
        pred = out.argmax(dim=1)
        mIoU = self.jaccard(pred, label)

        return {"loss" : loss, "mIoU" : mIoU}

