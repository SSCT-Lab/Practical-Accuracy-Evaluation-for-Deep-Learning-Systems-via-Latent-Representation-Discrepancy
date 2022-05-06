import torchvision
import torch
from torchvision import transforms
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import numpy as np


class MyCIFAR10(torchvision.datasets.cifar.CIFAR10):
    def __init__(self,
                 c_label: [],
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         download=download, train=train)
        if c_label is not None:
            self.targets = c_label


class MyCIFAR100(torchvision.datasets.cifar.CIFAR100):
    def __init__(self,
                 c_label: [],
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         download=download, train=train)
        if c_label is not None:
            self.targets = c_label


class MySVHN(torchvision.datasets.svhn.SVHN):
    def __init__(self,
                 c_label: [],
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         download=download, split=split)
        if c_label is not None:
            self.labels = c_label


class MyMNIST(torchvision.datasets.mnist.MNIST):
    def __init__(self,
                 c_label: [],
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         download=download, train=train)
        if c_label is not None:
            self.targets = c_label


class MyFashion(torchvision.datasets.mnist.FashionMNIST):
    def __init__(self,
                 c_label: [],
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         download=download, train=train)
        if c_label is not None:
            self.targets = c_label


# class CIFAR10_local_reload(torchvision.datasets.cifar.CIFAR10):
class CIFAR10_local_reload(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 label: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = data
        self.targets = label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR10_local_reload_without_label(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform)
        self.data = data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)

class CIFAR100_local_reload(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 label: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = data
        self.targets = label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)


class SVHN_local_reload(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 label: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = data
        self.targets = label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

class SVHN_local_reload_without_label(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)

class MNIST_local_reload(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 label: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = data
        self.targets = label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

class MNIST_local_reload_without_label(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)


class Fashion_local_reload(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 label: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = data
        self.targets = label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

class Fashion_local_reload_without_label(torchvision.datasets.VisionDataset):
    def __init__(self,
                 data: None,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)