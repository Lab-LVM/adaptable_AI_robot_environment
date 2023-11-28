import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from pathlib import Path

import torch
from torch import nn

from src_backup.cdan import get_model
from src.backbone.iresnet import get_arcface_backbone


class MyModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.layers = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]

    def forward(self, x):
        activations = []
        x = self.backbone.prelu(self.backbone.bn1(self.backbone.conv1(x)))
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations


def get_best_model(mode='arcface', base_path='log/best_weight/{}.pth'):
    model_path_dict = {'BSP': 'FACE_CDAN_BSP_BOTH', 'DAN': 'FACE_DAN_BOTH',
                       'BOTH': 'FACE_BOTH', 'FACE': 'FACE'}
    backbone = get_arcface_backbone('cpu')
    if mode != 'arcface':
        backbone = get_model(backbone, fc_dim=512, embed_dim=512, nclass=460, hidden_dim=1024,
                             pretrained_path=base_path.format(model_path_dict[mode])).backbone
    backbone.eval()
    return MyModel(backbone)


def img_preprocessing(img):
    transforms = A.Compose([
        A.SmallestMaxSize(112),
        A.CenterCrop(112, 112, p=1),
    ])
    img = ((np.transpose(transforms(image=np.array(img))['image'], (2, 0, 1)) / 255) - 0.5) / 0.5
    return img


def activation_based_map_f(activations):
    attention_map = []
    for activation in activations:
        img = activation.pow(2).mean(1).detach().numpy()[0, :, :, np.newaxis]
        resized_img = A.Resize(112, 112, 4)(image=img)['image']
        attention_map.append((resized_img, img))
    return attention_map


def show_example(img_path='iu_mask.jpg', mode='arcface', show=True):
    img = Image.open(img_path)
    img_resized = A.Resize(112, 112)(image=np.array(img))['image']
    img_np = np.array(img)
    img_np = img_preprocessing(img)
    input_img = torch.from_numpy(img_np).float().unsqueeze(0)
    model = get_best_model(mode)
    activations = model(input_img)
    attention_maps = activation_based_map_f(activations)
    if show:
        plt.imshow(img)
        plt.show()
        for attention_map in attention_maps:
            plt.figure(figsize=(16, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(img_resized, interpolation='bicubic')
            plt.imshow(attention_map[0], alpha=0.8, interpolation='bicubic')
            plt.subplot(1, 2, 2)
            plt.imshow(attention_map[1], interpolation='bicubic')
            plt.show()
    return [maps[0] for maps in attention_maps]


def compare_example(img_path, mode1='arcface', mode2='BSP', alpha=0.7, show=False):
    transforms = A.Compose([
        A.SmallestMaxSize(112),
        A.CenterCrop(112, 112, p=1),
    ])
    img = transforms(image=np.array(Image.open(img_path)))['image']
    attn1 = show_example(img_path=img_path, mode=mode1, show=False)
    attn2 = show_example(img_path=img_path, mode=mode2, show=False)
    plt.figure(figsize=(16, 6))
    plt.subplot(2, 5, 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    for i, attention_map in enumerate(zip(attn1, attn2)):
        plt.subplot(2, 5, 2 + i)
        plt.imshow(img, alpha=0.8)
        plt.imshow(attention_map[0], alpha=alpha, interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 5, 7 + i)
        plt.imshow(img, alpha=0.8)
        plt.imshow(attention_map[1], alpha=alpha, interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
    if show:
        plt.show()
    else:
        Path('result/attention_fig/').mkdir(exist_ok=True, parents=True)
        plt.savefig('result/attention_fig/{}_{}_{}_{}.jpg'.format(
            os.path.basename(img_path).split('.')[0], mode1, mode2, int(alpha*10)))
    plt.close('all')


def run(args):
    for mode in ['FACE', 'BSP', 'BOTH', 'DAN']:
        for image_path in ['iu.jpg', 'iu_mask1.jpg', 'iu_mask2.jpg', 'iu_mask3.jpg', 'iu_mask4.jpg']:
            for alpha in [0.8, 0.9]:
                print('mode: {}'.format(mode))
                print('alpha: {}'.format(alpha))
                compare_example(img_path='examples/{}'.format(image_path), mode1='arcface', mode2=mode, alpha=alpha)