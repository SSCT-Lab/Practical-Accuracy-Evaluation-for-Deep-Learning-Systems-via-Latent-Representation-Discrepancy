from __future__ import print_function
import argparse
from models.densenet import DenseNet3 as cifar_DenseNet3
from models.vgg import VGG as cifar_VGG
from models_mnist.vgg import VGG as mnist_VGG
from models_mnist.resnet import ResNet18 as mnist_ResNet18
import os
import torchvision
from torchvision import transforms
import torch.optim as optim
from utils import progress_bar
import torch.backends.cudnn as cudnn
import random
import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

params_map = {
    'rotation_range': (-10, 10),
    'brightness_range': (0.8, 1.3),
    'contrast_range': (0.8, 1.3),
    'shift_range': (0.05, 0.05)
}


# Training
def train(epoch, train_loader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def model_test(epoch, save_path, test_loader, net, criterion, best_acc, direct_save=False, last_epoch=None):
    # global best_acc
    net.eval()
    if not direct_save or (direct_save and epoch == last_epoch):
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total

    # Save checkpoint.
    if direct_save:
        print('Only save the model in the last epoch.')
        if epoch == last_epoch:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('pre_trained'):
                os.mkdir('pre_trained')
            torch.save(state, save_path)
    else:
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('pre_trained'):
                os.mkdir('pre_trained')
            torch.save(state, save_path)
            best_acc = acc
    return best_acc


def mnist_model(net_type, data_root, batch, lr, with_aug):
    num_classes, train_num = 10, 60000
    best_acc, start_epoch = 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch
    pre_trained_path = './pre_trained/' + net_type + '_mnist/'
    os.makedirs(pre_trained_path, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    aug_transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.ColorJitter(brightness=(0.8, 1.5), contrast=(0.8, 1.5)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.GaussianBlur(5, 5),
        transforms.RandomAffine(0, None, (0.7, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    # Data
    print('==> Preparing data..')

    if with_aug == 0:  # 原始训练集
        pre_trained_net = pre_trained_path + net_type + '_mnist.pth'
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=data_root, train=False, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
    else:  # 在原始训练集的基础上加一部分扩增训练集
        pre_trained_net = pre_trained_path + net_type + '_mnist_aug' + str(int(with_aug * 100)) + '.pth'
        aug_num = int(with_aug * train_num)
        print("Aug numer:", aug_num)
        sampler_list = list(range(train_num, train_num * 2))
        random.shuffle(sampler_list)
        sampler_list = list(range(train_num)) + sampler_list[:aug_num]  # 前 train_num 是原始训练集，后半部分是打乱了的 aug数据
        random.shuffle(sampler_list)
        train_sample_num = int(len(sampler_list) * 0.8)
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True) +
            torchvision.datasets.MNIST(root=data_root, train=True, transform=aug_transform_train, download=True),
            batch_size=batch, shuffle=False, sampler=sampler_list[:train_sample_num], num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True) +
            torchvision.datasets.MNIST(root=data_root, train=True, transform=aug_transform_train, download=True),
            batch_size=batch, shuffle=False, sampler=sampler_list[train_sample_num:], num_workers=1)

    # Model
    print('==> Building model..')
    if net_type == 'vgg11':
        net = mnist_VGG('VGG11')
    elif net_type == 'resnet18':
        net = mnist_ResNet18(num_c=10)

    net = net.to(device)
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + 50):
        train(epoch, train_loader, net, criterion, optimizer)
        best_acc = model_test(epoch, pre_trained_net, valid_loader, net, criterion, best_acc)
        scheduler.step()


def fashion_model(net_type, data_root, batch, lr, with_aug):
    num_classes, train_num = 10, 60000
    best_acc, start_epoch = 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch
    pre_trained_path = './pre_trained/' + net_type + '_fashion/'
    os.makedirs(pre_trained_path, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    aug_transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        # transforms.GaussianBlur(3, 5),
        transforms.RandomAffine(0, None, (0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    # Data
    print('==> Preparing data..')

    if with_aug == 0:  # 原始训练集
        pre_trained_net = pre_trained_path + net_type + '_fashion.pth'
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
    else:  # 在原始训练集的基础上加一部分扩增训练集
        pre_trained_net = pre_trained_path + net_type + '_fashion_aug' + str(int(with_aug * 100)) + '.pth'
        aug_num = int(with_aug * train_num)
        print("Aug numer:", aug_num)
        sampler_list = list(range(train_num, train_num * 2))
        random.shuffle(sampler_list)
        sampler_list = list(range(train_num)) + sampler_list[:aug_num]  # 前 train_num 是原始训练集，后半部分是打乱了的 aug数据
        random.shuffle(sampler_list)
        train_sample_num = int(len(sampler_list) * 0.8)
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=True) +
            torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=aug_transform_train, download=True),
            batch_size=batch, shuffle=False, sampler=sampler_list[:train_sample_num], num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=True) +
            torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=aug_transform_train, download=True),
            batch_size=batch, shuffle=False, sampler=sampler_list[train_sample_num:], num_workers=1)

    # Model
    print('==> Building model..')
    if net_type == 'vgg11':
        net = mnist_VGG('VGG11')
    elif net_type == 'resnet18':
        net = mnist_ResNet18(num_c=10)

    net = net.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + 50):
        train(epoch, train_loader, net, criterion, optimizer)
        best_acc = model_test(epoch, pre_trained_net, valid_loader, net, criterion, best_acc)
        scheduler.step()


def cifar10_model(net_type, data_root, batch, lr, with_aug):
    num_classes, train_num = 10, 50000
    best_acc, start_epoch = 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch
    pre_trained_path = './pre_trained/' + net_type + '_cifar10/'
    os.makedirs(pre_trained_path, exist_ok=True)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    aug_transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=params_map['rotation_range']),
        transforms.ColorJitter(brightness=params_map['brightness_range'], contrast=params_map['contrast_range']),
        transforms.RandomAffine(degrees=0, translate=params_map['shift_range']),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data
    print('==> Preparing data..')
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=data_root, train=False, transform=transform_test, download=True),
        batch_size=batch, shuffle=False, num_workers=1)
    if with_aug == 0:  # 原始训练集
        pre_trained_net = pre_trained_path + net_type + '_cifar10_aug0.pth'
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=data_root, train=True, transform=transform_train, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=data_root, train=False, transform=transform_test, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
    else:  # 在原始训练集的基础上加一部分扩增训练集
        pre_trained_net = pre_trained_path + net_type + '_cifar10_aug' + str(int(with_aug * 100)) + '.pth'
        aug_num = int(with_aug * train_num)
        print("Aug numer:", aug_num)
        sampler_list = list(range(train_num, train_num * 2))
        random.shuffle(sampler_list)
        sampler_list = list(range(train_num)) + sampler_list[:aug_num]  # 前 train_num 是原始训练集，后半部分是打乱了的 aug数据
        random.shuffle(sampler_list)
        train_sample_num = int(len(sampler_list) * 0.8)
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=data_root, train=True, transform=transform_train, download=True) +
            torchvision.datasets.CIFAR10(root=data_root, train=True, transform=aug_transform_train, download=True),
            batch_size=batch, shuffle=False, sampler=sampler_list[:train_sample_num], num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=data_root, train=True, transform=transform_train, download=True) +
            torchvision.datasets.CIFAR10(root=data_root, train=True, transform=aug_transform_train, download=True),
            batch_size=batch, shuffle=False, sampler=sampler_list[train_sample_num:], num_workers=1)

    # Model
    print('==> Building model..')
    if net_type == "densenet":
        net = cifar_DenseNet3(100, num_classes=num_classes)
    elif net_type == "vgg16":
        net = cifar_VGG('VGG16')

    net = net.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    epoch = 200
    for epoch in range(start_epoch, start_epoch + epoch):
        train(epoch, train_loader, net, criterion, optimizer)
        best_acc = model_test(epoch, pre_trained_net, valid_loader, net, criterion, best_acc)
        scheduler.step()


def svhn_model(net_type, data_root, batch, lr, with_aug):
    num_classes, train_num = 10, 73257
    best_acc, start_epoch = 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch
    pre_trained_path = './pre_trained/' + net_type + '_svhn/'
    os.makedirs(pre_trained_path, exist_ok=True)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
    ])
    aug_transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.ColorJitter(brightness=(0.7, 1.5), contrast=(0.7, 1.5)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.GaussianBlur(5, 5),
        transforms.RandomAffine(0, None, (0.8, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
    ])

    # Data
    print('==> Preparing data..')
    if with_aug == 0:  # 原始训练集
        pre_trained_net = pre_trained_path + net_type + '_svhn_aug0.pth'
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=data_root, split='train', transform=transform_train, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=data_root, split='test', transform=transform_test, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
    else:  # 在原始训练集的基础上加一部分扩增训练集
        pre_trained_net = pre_trained_path + net_type + '_svhn_aug' + str(int(with_aug * 100)) + '.pth'
        aug_num = int(with_aug * train_num)
        print("Aug numer:", aug_num)
        sampler_list = list(range(train_num, train_num * 2))
        random.shuffle(sampler_list)
        sampler_list = list(range(train_num)) + sampler_list[:aug_num]  # 前 train_num 是原始训练集，后半部分是打乱了的 aug数据
        random.shuffle(sampler_list)
        train_sample_num = int(len(sampler_list) * 0.8)
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=data_root, split='train', transform=transform_train, download=True) +
            torchvision.datasets.SVHN(root=data_root, split='train', transform=aug_transform_train, download=True),
            batch_size=batch, shuffle=False, sampler=sampler_list[:train_sample_num], num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=data_root, split='train', transform=transform_train, download=True) +
            torchvision.datasets.SVHN(root=data_root, split='train', transform=aug_transform_train, download=True),
            batch_size=batch, shuffle=False, sampler=sampler_list[train_sample_num:], num_workers=1)

    # Model
    print('==> Building model..')
    if net_type == "densenet":
        net = cifar_DenseNet3(100, num_classes=num_classes)
    elif net_type == "vgg16":
        net = cifar_VGG('VGG16')

    net = net.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch, train_loader, net, criterion, optimizer)
        best_acc = model_test(epoch, pre_trained_net, valid_loader, net, criterion, best_acc)
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch code: Train the models.')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--net_type', required=True, help='resnet34 | densenet | lenet1 | lenet5 | vgg11')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--with_aug', type=float, nargs='+', default=0.0, help='adding aug data for training')
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(args.gpu)
    data_root = os.path.expanduser(os.path.join("./data", str(args.dataset) + '-data'))

    for aug_rate in args.with_aug:
        if args.dataset == 'mnist':
            mnist_model(args.net_type, data_root, args.batch_size, args.lr, aug_rate)
        elif args.dataset == 'fashion':
            fashion_model(args.net_type, data_root, args.batch_size, args.lr, aug_rate)
        elif args.dataset == 'cifar10':
            cifar10_model(args.net_type, data_root, args.batch_size, args.lr, aug_rate)
        elif args.dataset == 'svhn':
            svhn_model(args.net_type, data_root, args.batch_size, args.lr, aug_rate)
