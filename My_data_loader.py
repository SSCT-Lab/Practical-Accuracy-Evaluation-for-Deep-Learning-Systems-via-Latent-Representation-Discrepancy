import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms, datasets
import os
import torchvision
from typing import Callable, Optional
import data_utils
import random


def get_CIFAR10_with_aug(batch_size, ori_TF, data_root, aug_TF=None, sampler=None, train_with_aug=False, test_with_aug=False, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        if train_with_aug:
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root=data_root, train=True, transform=ori_TF, download=True) +
                datasets.CIFAR10(root=data_root, train=True, transform=aug_TF, download=True),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root=data_root, train=True, transform=ori_TF, download=True),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        if test_with_aug:
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root=data_root, train=False, transform=ori_TF, download=True) +
                datasets.CIFAR10(root=data_root, train=False, transform=aug_TF, download=True),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root=data_root, train=False, transform=ori_TF, download=True),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_CIFAR100_with_aug(batch_size, ori_TF, data_root, aug_TF=None, sampler=None, train_with_aug=False, test_with_aug=False, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        if train_with_aug:
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(root=data_root, train=True, transform=ori_TF, download=True) +
                datasets.CIFAR100(root=data_root, train=True, transform=aug_TF, download=True),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(root=data_root, train=True, transform=ori_TF, download=True),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        if test_with_aug:
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(root=data_root, train=False, transform=ori_TF, download=True) +
                datasets.CIFAR100(root=data_root, train=False, transform=aug_TF, download=True),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(root=data_root, train=False, transform=ori_TF, download=True),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_SVHN_with_aug(batch_size, ori_TF, data_root, aug_TF=None, sampler=None, train_with_aug=False, test_with_aug=False, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        if train_with_aug:
            train_loader = torch.utils.data.DataLoader(
                datasets.SVHN(root=data_root, split='train', download=True, transform=ori_TF) +
                datasets.SVHN(root=data_root, split='train', download=True, transform=aug_TF),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.SVHN(root=data_root, split='train', download=True, transform=ori_TF),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)

    if val:
        if test_with_aug:
            test_loader = torch.utils.data.DataLoader(
                datasets.SVHN(root=data_root, split='test', download=True, transform=ori_TF) +
                datasets.SVHN(root=data_root, split='test', download=True, transform=aug_TF),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            test_loader = torch.utils.data.DataLoader(
                datasets.SVHN(root=data_root, split='test', download=True, transform=ori_TF),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_mnist_with_aug(batch_size, ori_TF, data_root, aug_TF=None, sampler=None, train_with_aug=False, test_with_aug=False, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        if train_with_aug:
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(root=data_root, train=True, download=True, transform=ori_TF) +
                datasets.MNIST(root=data_root, train=True, download=True, transform=aug_TF),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(root=data_root, train=True, download=True, transform=ori_TF),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)

    if val:
        if test_with_aug:
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(root=data_root, train=False, download=True, transform=ori_TF) +
                datasets.MNIST(root=data_root, train=False, download=True, transform=aug_TF),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(root=data_root, train=False, download=True, transform=ori_TF),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_fashion_with_aug(batch_size, ori_TF, data_root, aug_TF=None, sampler=None, train_with_aug=False, test_with_aug=False, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'fashion-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        if train_with_aug:
            train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(root=data_root, train=True, download=True, transform=ori_TF) +
                datasets.FashionMNIST(root=data_root, train=True, download=True, transform=aug_TF),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(root=data_root, train=True, download=True, transform=ori_TF),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)

    if val:
        if test_with_aug:
            test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(root=data_root, train=False, download=True, transform=ori_TF) +
                datasets.FashionMNIST(root=data_root, train=False, download=True, transform=aug_TF),
                batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
        else:
            test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(root=data_root, train=False, download=True, transform=ori_TF),
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getDataSet_with_aug(data_type, batch_size, ori_TF, dataroot, train_with_aug=False, test_with_aug=False, aug_TF=None, sampler=None):
    train_loader, test_loader = None, None
    if data_type == 'cifar10':
        train_loader, test_loader = get_CIFAR10_with_aug(batch_size, ori_TF, dataroot, aug_TF=aug_TF, sampler=sampler,
                                                         train_with_aug=train_with_aug, test_with_aug=test_with_aug,
                                                         num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = get_CIFAR100_with_aug(batch_size, ori_TF, dataroot, aug_TF=aug_TF, sampler=sampler,
                                                          train_with_aug=train_with_aug, test_with_aug=test_with_aug,
                                                          num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = get_SVHN_with_aug(batch_size, ori_TF, dataroot, aug_TF=aug_TF, sampler=sampler,
                                                      train_with_aug=train_with_aug, test_with_aug=test_with_aug,
                                                      num_workers=0)
    elif data_type == 'mnist':
        train_loader, test_loader = get_mnist_with_aug(batch_size, ori_TF, dataroot, aug_TF=aug_TF, sampler=sampler,
                                                       train_with_aug=train_with_aug, test_with_aug=test_with_aug,
                                                       num_workers=1)
    elif data_type == 'fashion':
        train_loader, test_loader = get_fashion_with_aug(batch_size, ori_TF, dataroot, aug_TF=aug_TF, sampler=sampler,
                                                         train_with_aug=train_with_aug, test_with_aug=test_with_aug,
                                                         num_workers=1)
    return train_loader, test_loader


def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):  # 没有改写 label 的
    test_loader = None
    if data_type == 'cifar10':
        train_loader, test_loader = get_CIFAR10_with_aug(batch_size=batch_size, ori_TF=input_TF, data_root=dataroot)
    elif data_type == 'svhn':
        train_loader, test_loader = get_SVHN_with_aug(batch_size=batch_size, ori_TF=input_TF, data_root=dataroot)
    elif data_type == 'cifar100':
        train_loader, test_loader = get_CIFAR100_with_aug(batch_size=batch_size, ori_TF=input_TF, data_root=dataroot)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'mnist':
        mnist = datasets.MNIST(root=dataroot, train=False, download=True, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'fashion':
        fashion = datasets.FashionMNIST(root=dataroot, train=False, download=True, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(fashion, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'emnist_letters':
        emnist = datasets.EMNIST(root=dataroot, train=False, download=True, transform=input_TF, split='letters')
        test_loader = torch.utils.data.DataLoader(emnist, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'kmnist':
        kmnist = datasets.KMNIST(root=dataroot, train=False, download=True, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(kmnist, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader


def get_OOD_Dataset(ID_data, batch_size, TF, dataroot, with_train=False):
    data = None
    dataroot_imagenet = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
    testsetout_imagenet = datasets.ImageFolder(dataroot_imagenet, transform=TF)
    dataroot_lsun = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
    testsetout_lsun = datasets.ImageFolder(dataroot_lsun, transform=TF)

    if ID_data == "cifar10":  # out_dist_list: ['svhn', 'imagenet_resize', 'lsun_resize']
        svhn_root = os.path.expanduser(os.path.join(dataroot, 'svhn-data'))
        data = datasets.SVHN(root=svhn_root, split='test', download=True,
                             transform=TF) + testsetout_imagenet + testsetout_lsun
        if with_train:
            data_root = os.path.expanduser(os.path.join(dataroot, 'cifar10-data'))
            data = data + datasets.CIFAR10(root=data_root, train=True, transform=TF, download=True)
    elif ID_data == "cifar100":  # out_dist_list: ['svhn', 'imagenet_resize', 'lsun_resize']
        data = datasets.SVHN(root=dataroot, split='test', download=True,
                             transform=TF) + testsetout_imagenet + testsetout_lsun
        if with_train:
            data_root = os.path.expanduser(os.path.join(dataroot, 'cifar100-data'))
            data = data + datasets.CIFAR100(root=data_root, train=True, transform=TF, download=True)
    elif ID_data == "svhn":  # out_dist_list: ['cifar10', 'imagenet_resize', 'lsun_resize']
        data = datasets.CIFAR10(root=dataroot, train=False, download=True,
                                transform=TF) + testsetout_imagenet + testsetout_lsun
        if with_train:
            data_root = os.path.expanduser(os.path.join(dataroot, 'svhn-data'))
            data = data + datasets.SVHN(root=data_root, split='train', download=True, transform=TF)
    elif ID_data == "mnist":  # out_dist_list: ['EMNIST-letters', 'KMNIST', 'Fashion']
        emnist = datasets.EMNIST(root=dataroot, train=False, download=True, transform=TF, split='letters')
        kmnist = datasets.KMNIST(root=dataroot, train=False, download=True, transform=TF)
        fashion = datasets.FashionMNIST(root=dataroot, train=False, download=True, transform=TF)
        data = emnist + kmnist + fashion
        if with_train:
            data_root = os.path.expanduser(os.path.join(dataroot, 'mnist-data'))
            data = data + datasets.MNIST(root=data_root, train=True, download=True, transform=TF)
    elif ID_data == "fashion":  # out_dist_list: ['EMNIST-letters', 'KMNIST', 'MNIST']
        emnist = datasets.EMNIST(root=dataroot, train=False, download=True, transform=TF, split='letters')
        kmnist = datasets.KMNIST(root=dataroot, train=False, download=True, transform=TF)
        mnist = datasets.MNIST(root=dataroot, train=False, download=True, transform=TF)
        data = emnist + kmnist + mnist
        if with_train:
            data_root = os.path.expanduser(os.path.join(dataroot, 'fashion-data'))
            data = data + datasets.FashionMNIST(root=data_root, train=True, download=True, transform=TF)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader


def get_My_CIFAR10(batch_size, ori_TF, data_root, c_label, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            data_utils.MyCIFAR10(root=data_root, train=True, transform=ori_TF, download=True, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            data_utils.MyCIFAR10(root=data_root, train=False, transform=ori_TF, download=True, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_My_CIFAR100(batch_size, ori_TF, data_root, c_label, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            data_utils.MyCIFAR100(root=data_root, train=True, transform=ori_TF, download=True, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            data_utils.MyCIFAR100(root=data_root, train=False, transform=ori_TF, download=True, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_My_SVHN(batch_size, ori_TF, data_root, c_label, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            data_utils.MySVHN(root=data_root, split='train', download=True, transform=ori_TF, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            data_utils.MySVHN(root=data_root, split='test', download=True, transform=ori_TF, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_My_MNIST(batch_size, ori_TF, data_root, c_label, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            data_utils.MyMNIST(root=data_root, train=True, transform=ori_TF, download=True, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            data_utils.MyMNIST(root=data_root, train=False, transform=ori_TF, download=True, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_My_Fashion(batch_size, ori_TF, data_root, c_label, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'fashion-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            data_utils.MyFashion(root=data_root, train=True, transform=ori_TF, download=True, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            data_utils.MyFashion(root=data_root, train=False, transform=ori_TF, download=True, c_label=c_label),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getTargetDataSet(data_type, batch_size, input_TF, dataroot, c_label):  # 改写了 label 的
    train_loader, test_loader = None, None
    if data_type == 'cifar10':
        train_loader, test_loader = get_My_CIFAR10(batch_size=batch_size, ori_TF=input_TF, data_root=dataroot, c_label=c_label)
    elif data_type == 'cifar100':
        train_loader, test_loader = get_My_CIFAR100(batch_size=batch_size, ori_TF=input_TF, data_root=dataroot, c_label=c_label)
    elif data_type == 'svhn':
        train_loader, test_loader = get_My_SVHN(batch_size=batch_size, ori_TF=input_TF, data_root=dataroot, c_label=c_label)
    elif data_type == 'mnist':
        train_loader, test_loader = get_My_MNIST(batch_size=batch_size, ori_TF=input_TF, data_root=dataroot, c_label=c_label)
    elif data_type == 'fashion':
        train_loader, test_loader = get_My_Fashion(batch_size=batch_size, ori_TF=input_TF, data_root=dataroot, c_label=c_label)
    return train_loader, test_loader

