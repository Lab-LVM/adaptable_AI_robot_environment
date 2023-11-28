import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, List

import sklearn
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.backbone.models import get_model
from src.face_utils.verification import evaluate
from src.system.bsp import BSP_DANN, BSP_CDAN_E
from src.system.cdan import CDAN, CDAN_E
from src.system.dann import DANN
from src.system.mstn import MSTN
from src.system.shot import SHOT
from src.system.source_only import DABase
from src.system.trans_da import TransDA


def FaceDAWrapper(base_class):
    class FaceSystem(base_class):
        def __init__(self, *args, unmasked_init: dict = None, verifier_init: dict = None,
                     verifier_fc_init: str = None, ensemble_init: list = None, **kwargs):
            super(FaceSystem, self).__init__(*args, **kwargs)
            self.base_class_name = self.__class__.__bases__[0].__name__
            self.unmasked_init = unmasked_init
            self.verifier_init = verifier_init
            self.verifier_fc_init = verifier_fc_init
            self.ensemble_init = ensemble_init

        def forward(self, x):
            return self.backbone(x)
            # return F.normalize(self.backbone(x))
            # return F.normalize(self.bottleneck(self.backbone(x)), dim=-1)

        def validation_step(self, batch, batch_idx, dataloader_idx=None):
            with torch.no_grad():
                img1, img2, label = batch
                feat1, feat1_flip = self(img1).cpu(), self(torch.flip(img1, [3])).cpu()
                feat2, feat2_flip = self(img2).cpu(), self(torch.flip(img2, [3])).cpu()
            return feat1, feat1_flip, feat2, feat2_flip, label

        def validation_epoch_end(self, results: List[Any]) -> None:
            acc_dict = {}
            for idx, batchs in enumerate(results):
                embed = [np.concatenate([batch[i].cpu().numpy() for batch in batchs], axis=0) for i in range(5)]
                embed1 = sklearn.preprocessing.normalize(embed[0] + embed[1])
                embed2 = sklearn.preprocessing.normalize(embed[2] + embed[3])
                label = embed[4]
                _, _, accuracy, _, _, far = evaluate(embed1, embed2, label)
                acc, std = np.mean(accuracy), np.std(accuracy)
                acc_dict[f'valid/data{idx}_acc'] = acc
                # self.logger.experiment[0][f'test/verification/data{idx}_acc'].log(acc)
                # self.print(f'test/verification/data{idx}_acc: {acc}')
            # acc_dict["trainer/global_step"] = self.current_epoch
            self.log_dict(acc_dict)

        def test_step(self, batch, batch_idx, dataloader_idx=None):
            with torch.no_grad():
                img1, img2, label = batch
                feat1, feat1_flip = self(img1).cpu(), self(torch.flip(img1, [3])).cpu()
                feat2, feat2_flip = self(img2).cpu(), self(torch.flip(img2, [3])).cpu()
            return feat1, feat1_flip, feat2, feat2_flip, label

        def test_epoch_end(self, results: List[Any]) -> None:
            acc_dict = {}
            for idx, batchs in enumerate(results):
                embed = [np.concatenate([batch[i].cpu().numpy() for batch in batchs], axis=0) for i in range(5)]
                embed1 = sklearn.preprocessing.normalize(embed[0] + embed[1])
                embed2 = sklearn.preprocessing.normalize(embed[2] + embed[3])
                label = embed[4]
                _, _, accuracy, _, _, far = evaluate(embed1, embed2, label)
                acc, std = np.mean(accuracy), np.std(accuracy)
                acc_dict[f'valid/data{idx}_acc'] = acc
                # self.logger.experiment[0][f'test/verification/data{idx}_acc'].log(acc)
                # self.print(f'test/verification/data{idx}_acc: {acc}')
            # acc_dict["trainer/global_step"] = self.current_epoch
            self.log_dict(acc_dict)

        def on_predict_start(self) -> None:
            if self.ensemble_init:
                mask_ensemble = nn.ModuleList(get_model(**init).to(self.device) for init in self.ensemble_init)
                self.mask = lambda x: sum(f(x) for f in mask_ensemble)
                self.logger.experiment[0]['Method'] = '+'.join(item['pretrained_path'].split('/')[1].split('_')[0] for item in self.ensemble_init)
            else:
                self.mask = self.backbone
                self.logger.experiment[0]['Method'] = self.base_class_name

            self.unmask = get_model(**self.unmasked_init).to(self.device)
            self.verifier = get_model(**self.verifier_init).to(self.device)
            self.fc = nn.Linear(self.verifier.out_channels, 2)
            self.fc.load_state_dict(torch.load(self.verifier_fc_init, map_location='cpu'))
            self.fc.to(self.device)

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            if dataloader_idx < 2:
                return self.gallery_step(batch, dataloader_idx)
            elif dataloader_idx < 4:
                return self.prove_step(batch)
            else:
                return self.verification_step(batch)

        def verification_step(self, batch):
            img1, img2, label = batch
            wear_mask = torch.logical_or(self.is_masked(img1), self.is_masked(img2))
            feat1, feat1_flip = self.get_masked_unmasked_features(img1, wear_mask)
            feat2, feat2_flip = self.get_masked_unmasked_features(img2, wear_mask)
            label = torch.cat([label[wear_mask], label[wear_mask == 0]])
            return feat1, feat1_flip, feat2, feat2_flip, label

        def gallery_step(self, batch, dataloader_idx):
            img, label = batch
            feat, feat_flip = self.get_features(img, self.unmask if dataloader_idx == 3 else self.mask)
            return feat, feat_flip, label

        def prove_step(self, batch):
            img, label = batch
            wear_mask = self.is_masked(img)
            return *self.get_all_features(img, wear_mask), label[wear_mask], label[wear_mask == 0]

        def get_masked_unmasked_features(self, img, wear_mask):
            feat_masked, feat_masked_flip, feat_unmasked, feat_unmasked_flip = self.get_all_features(img, wear_mask)
            return torch.cat([feat_masked, feat_unmasked]), torch.cat([feat_masked_flip, feat_unmasked_flip])

        def get_all_features(self, img, wear_mask):
            feat_masked, feat_masked_flip = self.get_features(img[wear_mask], self.mask)
            feat_unmasked, feat_unmasked_flip = self.get_features(img[wear_mask == 0], self.unmask)
            return feat_masked, feat_masked_flip, feat_unmasked, feat_unmasked_flip

        def get_features(self, img, extractor):
            if len(img) == 0:
                return torch.tensor([]).to(img.device), torch.tensor([]).to(img.device)
            return extractor(img), extractor(torch.flip(img, [3]))

        def is_masked(self, img1):
            return self.fc(self.verifier(img1)).max(dim=1)[1] == 0

        def on_predict_epoch_end(self, results: List[Any]) -> None:
            self.identification_epoch_end(results[:4])
            self.verification_epoch_end(results[4:])

        def identification_epoch_end(self, results):
            unmask_gallery, mask_gallery = self.compute_embed(results[0])[0], self.compute_embed(results[1])[0]

            for num_identity in [*list(range(0, len(unmask_gallery), 10)), len(unmask_gallery)]:
                unmask_acc = self.compute_identification_acc(results[2], num_identity, np.copy(mask_gallery),
                                                             np.copy(unmask_gallery))
                mask_acc = self.compute_identification_acc(results[3], num_identity, np.copy(mask_gallery),
                                                           np.copy(unmask_gallery))
                self.logger.experiment[0][f'predict/identification/unmask_acc'].log(unmask_acc)
                self.logger.experiment[0][f'predict/identification/mask_acc'].log(mask_acc)
                self.print(f'predict/verification/unmask_acc: {unmask_acc}')
                self.print(f'predict/verification/mask_acc: {mask_acc}')

        def compute_identification_acc(self, result, num_identity, mask_gallery, unmask_gallery):
            embed = self.cat_batchs(result, 6)
            mask_prove = sklearn.preprocessing.normalize(embed[0] + embed[1])
            unmask_prove = sklearn.preprocessing.normalize(embed[2] + embed[3])
            mask_label, unmask_label = embed[4], embed[5]
            mask_prove, mask_gallery, mask_label = self.apply_identity_mask(mask_prove, mask_gallery, mask_label,
                                                                            num_identity)
            unmask_prove, unmask_gallery, unmask_label = self.apply_identity_mask(unmask_prove, unmask_gallery,
                                                                                  unmask_label, num_identity)

            mask_sum = ((mask_prove @ mask_gallery.transpose(1, 0)).argmax(axis=1) == mask_label).astype(np.float).sum()
            unmask_sum = ((unmask_prove @ unmask_gallery.transpose(1, 0)).argmax(axis=1) == unmask_label).astype(
                np.float).sum()

            return (mask_sum + unmask_sum) / (len(mask_prove) + len(unmask_prove))

        def apply_identity_mask(self, prove_set, gallery_set, label, num_identity):
            num_classes = len(gallery_set)
            one_hot_label = np.eye(num_classes)[label]
            identity_mask = np.zeros(num_classes)
            identity_mask[:num_identity] = 1
            identity_mask = (one_hot_label @ identity_mask.reshape(1, -1).transpose(1, 0)).reshape(-1) == 1
            label = label[identity_mask]
            prove_set = prove_set[identity_mask]
            gallery_set[num_identity:] = 0

            return prove_set, gallery_set, label

        def compute_embed(self, results):
            feat, feat_flip, label = self.cat_batchs(results, 3)
            embed = sklearn.preprocessing.normalize(feat + feat_flip)
            return embed, label

        def cat_batchs(self, result, col_len):
            return [np.concatenate([batch[i].detach().cpu().numpy() for batch in result if len(batch[i])], axis=0) for i
                    in range(col_len)]

        def verification_epoch_end(self, results):
            for idx, batchs in enumerate(results):
                embed = self.cat_batchs(batchs, 5)
                embed1 = sklearn.preprocessing.normalize(embed[0] + embed[1])
                embed2 = sklearn.preprocessing.normalize(embed[2] + embed[3])
                label = embed[4]
                _, _, accuracy, _, _, far = evaluate(embed1, embed2, label)
                acc, std = np.mean(accuracy), np.std(accuracy)
                self.logger.experiment[0][f'predict/verification/data{idx}_acc'].log(acc)
                self.print(f'predict/verification/data{idx}_acc: {acc}')

        def on_save_checkpoint(self, checkpoint):
            save_path = os.path.join('pretrained', self.base_class_name + '_' + self.backbone.__class__.__name__)
            Path(save_path).mkdir(exist_ok=True, parents=True)

            with open(os.path.join(save_path, self.trainer.datamodule.pair + '.ckpt'), 'wb') as f:
                torch.save(nn.Sequential(OrderedDict([('backbone', self.backbone), ('bottleneck', self.bottleneck), ('fc', self.fc)])).state_dict(), f)

            with open(os.path.join(save_path, self.trainer.datamodule.pair + '_backbone' + '.ckpt'), 'wb') as f:
                torch.save(self.backbone.state_dict(), f)

    return FaceSystem


class NO_SHOT(SHOT):
    def make_pseudo_label(self, model, classifier) -> None:
        import torch.nn.functional as F
        model.eval()
        classifier.eval()
        with torch.no_grad():
            embed, p, ys = [], [], []
            # Todo: Changed
            for x, y in self.trainer.datamodule.train_dataloader():
                embed.append(model(x.to(self.device)))
                p.append(F.softmax(classifier(embed[-1]), dim=1))
                ys.append(y.to(self.device))
            embed, p, ys = torch.cat(embed, dim=0), torch.cat(p, dim=0), torch.cat(ys, dim=0)
            pseudo_label = p.argmax(dim=1)
            print('acc: {:5.2f}'.format((p.argmax(dim=1) == ys).float().mean()*100))

            tgt_train = self.trainer.datamodule.train_dataloader().dataset
            tgt_train.samples = [(tgt_train.samples[i][0], pseudo_label[i].item()) for i in range(len(pseudo_label))]

        model.train()
        classifier.train()

        pass


Face_DABase = FaceDAWrapper(DABase)
Face_DANN = FaceDAWrapper(DANN)
Face_CDAN = FaceDAWrapper(CDAN)
Face_CDAN_E = FaceDAWrapper(CDAN_E)
Face_BSP_DANN = FaceDAWrapper(BSP_DANN)
Face_BSP_CDAN_E = FaceDAWrapper(BSP_CDAN_E)
Face_MSTN = FaceDAWrapper(MSTN)
Face_SHOT = FaceDAWrapper(SHOT)
Face_NO_SHOT = FaceDAWrapper(NO_SHOT)
Face_TransDA = FaceDAWrapper(TransDA)

