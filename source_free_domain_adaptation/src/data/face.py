import glob
import json
import os
import random
from PIL import Image

from typing import Tuple, List, Dict, Optional, Callable
from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class MyImageFolder(ImageFolder):
    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = None):
        self.split = split
        super(MyImageFolder, self).__init__(root=root, transform=transform)

    def find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        with open(dir + '/../split_class_dict.json', 'rt', encoding='utf-8') as f:
            json_list = list(filter(lambda x: len(glob.glob(os.path.join(dir, x[0], '*'))), json.load(f)[self.split]))
            classes = [name for (name, label) in json_list]
            class_to_idx = {name: label for (name, label) in json_list}
        return classes, class_to_idx


class MyImageFolderIdx(MyImageFolder):
    def __getitem__(self, item):
        img, label = super(MyImageFolderIdx, self).__getitem__(item)
        return img, label, item


class MyVerification(Dataset):
    def __init__(self,
                 root: str,
                 pairs: tuple = ('Unmasked', 'Masked'),
                 transform: Optional[Callable] = None,
                 split: str = 'test',
                 pair_len: int = 3000
                 ):
        super(MyVerification, self).__init__()
        self.root = root
        self.pairs = pairs
        self.split = split
        self.transform = transform
        self.pair_len = pair_len
        self.samples = self.get_samples()

    def get_samples(self):
        classes, class_to_idx = self.find_classes()
        class_to_instance = self.get_class_to_instance(classes)
        return self.get_pair(classes, class_to_instance)

    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        with open(os.path.join(self.root, 'split_class_dict.json'), 'rt', encoding='utf-8') as f:
            json_list = json.load(f)[self.split]
            classes = [name for (name, label) in json_list]
            class_to_idx = {name: label for (name, label) in json_list}
        return classes, class_to_idx

    def get_class_to_instance(self, classes):
        return [{name: glob.glob(os.path.join(self.root, self.pairs[idx], name, '*')) for name in classes} for idx in
                range(2)]

    def get_pair(self, classes: List, class_to_instance: Tuple[Dict[str, List]]):
        dataset = []
        num_class = len(classes)
        class1, class2 = class_to_instance
        while len(dataset) < self.pair_len:
            name = classes[random.randrange(0, num_class)]
            if len(class1[name]) and len(class2[name]):
                dataset.append((random.choice(class1[name]), random.choice(class2[name]), 1))

        while len(dataset) < self.pair_len * 2:
            pairs = random.choice(list(class1.keys())), random.choice(list(class2.keys()))
            while pairs[0] == pairs[1]:
                pairs = random.choice(list(class1.keys())), random.choice(list(class2.keys()))
            if len(class1[pairs[0]]) and len(class2[pairs[1]]):
                dataset.append((random.choice(class1[pairs[0]]), random.choice(class2[pairs[1]]), 0))

        random.shuffle(dataset)
        return dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1, img2, label = self.samples[idx]
        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


class MyIdenficiation(Dataset):
    def __init__(self, root: str, split: str = 'prove_set', transform: Optional[Callable] = None):
        super(MyIdenficiation, self).__init__()
        self.ds, self.label = self.read_ds(root, split)
        self.classes = list(set(self.label))
        self.transform = transform

    def read_ds(self, root, split):
        ds = []
        labels = []
        with open(os.path.join(root, split+'.txt'), 'rt') as f:
            for line in f.readlines():
                idx, label = line.strip('\n').split(' ')
                ds.append(os.path.join(root, split, idx+'.jpg'))
                labels.append(int(label))
        return ds, labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        img, label = self.ds[item], self.label[item]

        if self.transform:
            img = self.transform(Image.open(img))

        return img, label


def get_trasnforms(resize, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2),
    ])
    test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train, test


class SourceOnly(LightningDataModule):
    def __init__(self, dataset_name: str, pair: str, size: tuple, data_root: str, test_data_root: str,
                 batch_size: int, num_workers: int, valid_ratio: float, test_pair:str=None):
        super(SourceOnly, self).__init__()

        src, tgt = pair.split('_')

        self.pair = pair
        self.test_pair = test_pair.split('_') if test_pair else ('Unmasked', 'Masked')
        self.src = src
        self.tgt = tgt
        self.root = f'{data_root}/{dataset_name}'
        self.test_root = f'{test_data_root}/{dataset_name}'
        self.src_root = f'{self.root}/{src}'
        self.tgt_root = f'{self.root}/{tgt}'
        self.dataset = MyImageFolder
        self.ver_dataset = MyVerification
        self.id_dataset = MyIdenficiation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio

        self.num_step = None
        self.num_classes = None
        self.train_transform, self.test_transform = get_trasnforms(size)
        self.prepare_data()

    def prepare_data(self) -> None:
        src_ds = self.dataset(root=self.src_root, split='train')
        tgt_ds = self.dataset(root=self.tgt_root, split='train')

        self.num_classes = len(src_ds.classes)
        self.num_step = int(len(src_ds) * (1 - self.valid_ratio)) // self.batch_size

        print('-' * 50)
        print('Face Source Only Dataset')
        print('* {} dataset number of class: {}'.format(self.src, self.num_classes))
        print('* {} dataset number of class: {}'.format(self.tgt, len(tgt_ds.classes)))
        print('* {} train dataset len: {}'.format(self.src, len(src_ds) * (1 - self.valid_ratio)))
        print('* {} valid dataset len: {}'.format(self.src, len(src_ds) * self.valid_ratio))
        print('-' * 50)

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            self.src_ds = self.dataset(self.src_root, 'train', self.train_transform)
            self.train_src_ds, self.valid_src_ds = self.split_train_valid(self.src_ds)
            self.train_tgt_ds = self.dataset(self.tgt_root, 'train', self.train_transform)
            self.test_tgt_ds = self.dataset(self.tgt_root, transform=self.test_transform)
            self.u_u = self.ver_dataset(self.root, (self.test_pair[0], self.test_pair[0]), self.test_transform)
            self.u_m = self.ver_dataset(self.root, (self.test_pair[0], self.test_pair[1]), self.test_transform)
            self.m_m = self.ver_dataset(self.root, (self.test_pair[1], self.test_pair[1]), self.test_transform)

        elif stage in (None, 'test', 'predict'):
            self.u_u = self.ver_dataset(self.root, (self.test_pair[0], self.test_pair[0]), self.test_transform)
            self.u_m = self.ver_dataset(self.root, (self.test_pair[0], self.test_pair[1]), self.test_transform)
            self.m_m = self.ver_dataset(self.root, (self.test_pair[1], self.test_pair[1]), self.test_transform)
            self.u_g = self.id_dataset(os.path.join(self.test_root, 'Unmasked'), 'gallery_set', self.test_transform)
            self.u_p = self.id_dataset(os.path.join(self.test_root, 'Unmasked'), 'prove_set', self.test_transform)
            self.m_g = self.id_dataset(os.path.join(self.test_root, 'Masked'), 'gallery_set', self.test_transform)
            self.m_p = self.id_dataset(os.path.join(self.test_root, 'Masked'), 'prove_set', self.test_transform)

    def split_train_valid(self, ds):
        ds_len = len(ds)
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        return random_split(ds, [train_ds_len, valid_ds_len])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_src_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.valid_src_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(self.u_u, batch_size=self.batch_size * 2, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.u_m, batch_size=self.batch_size * 2, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.m_m, batch_size=self.batch_size * 2, shuffle=False, num_workers=self.num_workers),
        ]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(self.u_u, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.u_m, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.m_m, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
        ]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(self.u_g, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.m_g, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.u_p, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.m_p, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.u_u, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.u_m, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.m_m, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
        ]


class DomainAdaptation(SourceOnly):
    def __init__(self, *args, num_step_mode: str = 'max', **kwargs):
        self.num_step_mode = num_step_mode
        super(DomainAdaptation, self).__init__(*args, **kwargs)

    def prepare_data(self) -> None:
        src_ds = self.dataset(root=self.src_root, split='train')
        tgt_ds = self.dataset(root=self.tgt_root, split='train')

        self.num_classes = len(src_ds.classes)
        ds_len = max(len(src_ds), len(tgt_ds)) if self.num_step_mode == 'max' else min(len(src_ds), len(tgt_ds))
        self.num_step = ds_len // self.batch_size

        print('-' * 50)
        print('Face Domain Adaptation Dataset')
        print('* {} dataset number of class: {}'.format(self.src, self.num_classes))
        print('* {} dataset number of class: {}'.format(self.tgt, len(tgt_ds.classes)))
        print('* {} train dataset len: {}'.format(self.src, len(src_ds)))
        print('* {} train dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('* {} valid dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('-' * 50)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        src = DataLoader(self.src_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        tgt = DataLoader(self.train_tgt_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        return [src, tgt]

    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.test_tgt_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class SourceFreeDomainAdaptation(DomainAdaptation):
    def __init__(self, *args, return_idx: bool = False, **kwargs):
        self.train_dataset = MyImageFolderIdx if return_idx else MyImageFolder
        super(SourceFreeDomainAdaptation, self).__init__(*args, **kwargs)

    def prepare_data(self) -> None:
        src_ds = self.dataset(root=self.src_root, split='train')
        tgt_ds = self.dataset(root=self.tgt_root, split='train')

        self.num_classes = len(src_ds.classes)
        self.num_step = len(tgt_ds) // self.batch_size

        print('-' * 50)
        print('Face Source Free Domain Adaptation Dataset')
        print('* {} dataset number of class: {}'.format(self.src, self.num_classes))
        print('* {} dataset number of class: {}'.format(self.tgt, len(tgt_ds.classes)))
        print('* {} train dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('* {} valid dataset len: {}'.format(self.tgt, len(tgt_ds)))
        print('-' * 50)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.train_tgt_ds = self.train_dataset(self.tgt_root, 'train', self.train_transform)
        return DataLoader(self.train_tgt_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)