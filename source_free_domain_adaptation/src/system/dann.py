import copy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import instantiate_class

from src.common_module import DomainClassifier
from src.face_utils.arcface import arcFace
from src.system.source_only import DABase


class DANN(DABase):
    def __init__(self, *args, hidden_dim: int = 1024, gamma: int = 10, source_only_path=None, **kwargs):
        super(DANN, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.dc = DomainClassifier(kwargs['embed_dim'], hidden_dim)
        self.criterion_dc = nn.CrossEntropyLoss()
        self.source_only_path = source_only_path
        self.bottleneck2 = copy.deepcopy(self.bottleneck)
        # self.layer_norm = nn.LayerNorm(self.backbone.out_channels)
        # self.fc = arcFace(kwargs['embed_dim'], nclass=kwargs['num_classes'])

    def on_fit_start(self) -> None:
        # Todo: Changed
        if self.source_only_path:
            if self.source_only_path.endswith('ckpt'):
                weight_path = self.source_only_path
            else:
                weight_path = os.path.join(self.source_only_path, 'Unmasked+Synthetic_Masked.ckpt')
                # weight_path = os.path.join(self.source_only_path, self.trainer.datamodule.pair + '.ckpt')
            self.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
            self.bottleneck2.load_state_dict(self.bottleneck.state_dict())
        self.to(memory_format=torch.channels_last)

    def on_after_batch_transfer(self, batch, *args, **kwargs):
        if isinstance(batch, list):
            if isinstance(batch[0], list):
                return [[b.to(memory_format=torch.channels_last) if b.dim() > 3 else b for b in bb] for bb in batch]
            else:
                return [b.to(memory_format=torch.channels_last) if b.dim() > 3 else b for b in batch]
        else:
            return batch.to(memory_format=torch.channels_last)

    def training_step(self, batch, batch_idx, optimizer_idx=None, child_compute_already=None):
        (x_s, y_s), (x_t, y_t) = batch
        if child_compute_already:
            (embed_s, y_hat_s), (embed_t, y_hat_t) = child_compute_already
        else:
            # embed_s, y_hat_s = self.get_feature(x_s, mask=False, label=y_s)
            # embed_t, y_hat_t = self.get_feature(x_t, mask=True, label=y_t)
            embed_s, y_hat_s = self.get_feature(x_s, mask=False)
            embed_t, y_hat_t = self.get_feature(x_t, mask=True)

        loss_dc = self.compute_dc_loss(embed_s, embed_t, y_hat_s, y_hat_t)
        # loss_cls = self.criterion(y_hat_s, y_s)
        loss_cls = self.criterion(y_hat_s, y_s) + self.criterion(y_hat_t, y_t)
        loss = loss_cls+loss_dc

        metric = self.train_metric(y_hat_s, y_s)
        self.log_dict({f'train/loss': loss})
        self.log_dict(metric)
        return loss

    def compute_dc_loss(self, embed_s, embed_t, y_hat_s, y_hat_t):
        y_hat_dc = self.dc(torch.cat([embed_s, embed_t]), self.get_alpha())
        y_dc = torch.cat([torch.zeros_like(y_hat_s[:, 0]), torch.ones_like(y_hat_t[:, 0])]).long()
        loss_dc = self.criterion_dc(y_hat_dc, y_dc)
        return loss_dc

    def get_feature(self, x, domain=None, mask=None, label=None):
        feature = self.backbone(x)
        # feature = self.layer_norm(feature)
        # feature = F.normalize(feature)
        if mask:
            embed = self.bottleneck(feature)
        else:
            embed = self.bottleneck2(feature)
        if label is not None:
            y_hat = self.fc(embed, label)
        else:
            y_hat = self.fc(embed)
        return embed, y_hat

    def get_alpha(self):
        # return 1
        return 2. / (1. + np.exp(-self.gamma * self.global_step / (self.num_step * self.max_epochs))) - 1

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * .1},
            # {'params': self.layer_norm.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 1},
            {'params': self.bottleneck.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 1},
            {'params': self.bottleneck2.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 1},
            {'params': self.fc.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 1},
            {'params': self.dc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}