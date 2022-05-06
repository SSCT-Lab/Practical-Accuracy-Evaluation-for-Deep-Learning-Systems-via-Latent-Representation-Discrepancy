import argparse
import random

import torch
import os
import numpy as np
import math

from models.densenet import DenseNet3 as cifar_DenseNet3
from models.vgg import VGG as cifar_VGG
from models_mnist.vgg import VGG as mnist_VGG
from models_mnist.resnet import ResNet18 as mnist_ResNet18
import data_utils

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import foolbox as fb
import matplotlib.pyplot as plt
from fid_score import *
from inception_score import *
from aug_utils import *
import selection
import lib
import metrics
from selection import *
import time
import sys
from datetime import datetime

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class Generator_gan(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator_gan, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Generator_dcgan(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator_dcgan, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

def get_dataloader(dataset, data_root, batch):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        # images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=100)
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=True, transform=train_transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=False, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        bounds = (-1, 1)
        preprocessing = dict(mean=(0.5,), std=(0.5,), axis=-1)

    elif dataset == 'fashion':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=data_root, train=False, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        bounds = (-1, 1)
        preprocessing = dict(mean=(0.5,), std=(0.5,))

    elif dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='train', transform=transform_train, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='test', transform=transform_test, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        bounds = (0, 255)
        preprocessing = dict(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614], axis=-3)

    elif dataset == 'cifar10':
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
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=True, transform=transform_train, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=False, transform=transform_test, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        bounds = (0, 255)
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)

    return train_loader, valid_loader

def get_grad_dataloader(dataset, data_root, batch):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        # images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=100)
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=True, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=False, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        bounds = (-1, 1)
        preprocessing = dict(mean=(0.5,), std=(0.5,), axis=-1)

    elif dataset == 'fashion':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=data_root, train=False, transform=transform, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        bounds = (-1, 1)
        preprocessing = dict(mean=(0.5,), std=(0.5,))

    elif dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='train', transform=transform_train, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='test', transform=transform_test, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        bounds = (0, 255)
        preprocessing = dict(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614], axis=-3)

    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=True, transform=transform_train, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=False, transform=transform_test, download=True),
            batch_size=batch, shuffle=False, num_workers=1)
        bounds = (0, 255)
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)

    return train_loader, valid_loader, bounds, preprocessing

def plot_clean_and_adver(adver_example, adver_target, clean_example, clean_target, attack_name, dataset):
    n_cols = 2
    n_rows = 5
    cnt = 1
    cnt1 = 1
    plt.figure(figsize=(4 * n_rows, 2 * n_cols))
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(n_cols):
            for j in range(n_rows):
                plt.subplot(n_cols, n_rows * 2, cnt1)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.ylabel(attack_name, size=15)
                plt.title("{} -> {}".format(clean_target[cnt - 1], adver_target[cnt - 1]))
                plt.imshow(clean_example[cnt - 1].reshape(28, 28).to('cpu').detach().numpy(), cmap='gray')
                plt.subplot(n_cols, n_rows * 2, cnt1 + 1)
                plt.xticks([])
                plt.yticks([])
                # plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
                plt.imshow(adver_example[cnt - 1].reshape(28, 28).to('cpu').detach().numpy(), cmap='gray')
                cnt = cnt + 1
                cnt1 = cnt1 + 2
    elif dataset == 'cifar10' or dataset == 'svhn':
        for i in range(n_cols):
            for j in range(n_rows):
                plt.subplot(n_cols, n_rows * 2, cnt1)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.ylabel(attack_name, size=15)
                plt.title("{} -> {}".format(clean_target[cnt - 1], adver_target[cnt - 1]))
                img = clean_example[cnt - 1].to('cpu').detach().numpy()
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                plt.imshow(img)
                plt.subplot(n_cols, n_rows * 2, cnt1 + 1)
                plt.xticks([])
                plt.yticks([])
                plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
                img = adver_example[cnt - 1].to('cpu').detach().numpy()
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                plt.imshow(img)
                # plt.imshow(adver_example[cnt - 1].reshape(32, 32).to('cpu').detach().numpy())
                cnt = cnt + 1
                cnt1 = cnt1 + 2
    plt.show()
    print('\n')

def grad_generation(dataloader, fmodel, grad_attack, saveroot, attack_method, dataset):
    acc, rob_acc = [], []
    adv_png_dataroot = os.path.expanduser(os.path.join(saveroot) + 'adv-png/')
    np_dataroot = os.path.expanduser(os.path.join(saveroot) + 'adv-np/')
    os.makedirs(adv_png_dataroot, exist_ok=True)
    os.makedirs(np_dataroot, exist_ok=True)
    save_adv_image, save_label, save_clean_image = [], [], []
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"-----------{batch_idx}------------")
        images, labels = images.cuda(), labels.cuda()
        clean_acc = fb.accuracy(fmodel, images, labels)
        print(f"clean accuracy:  {clean_acc * 100:.1f} %")
        acc.append(clean_acc)
        # 一个简陋版本的步幅控制
        if batch_idx < 10:
            # epsilon = 1
            epsilon = random.random()
            if dataset == 'mnist':
                weight = 0.5
            elif dataset == 'fashion':
                weight = 0.5
            elif dataset == 'svhn':
                weight = 0.04
            elif dataset == 'cifar10':
                weight = 0.04
            if batch_idx == 0:
                print('batch 0-9 weight:', weight)
            raw_advs, clipped_advs, success = grad_attack(fmodel, images, labels, epsilons=epsilon * weight)
        elif batch_idx < 20:
            # epsilon = 1
            epsilon = random.random()
            if dataset == 'mnist':
                weight = 0.1
            elif dataset == 'fashion':
                weight = 0.1
            elif dataset == 'svhn':
                weight = 0.3
            elif dataset == 'cifar10':
                weight = 0.2
            if batch_idx == 10:
                print('batch 10-19 weight:', weight)
            raw_advs, clipped_advs, success = grad_attack(fmodel, images, labels, epsilons=epsilon * weight)
        # elif batch_idx < 30:
        #     epsilon = random.random()
        #     weight = 0.1
        #     raw_advs, clipped_advs, success = grad_attack(fmodel, images, labels, epsilons=epsilon * weight)
        else:
            break
        robust_accuracy = 1 - success.float().mean(axis=-1)
        print(f"robust accuracy:  {robust_accuracy.cpu() * 100:.1f} %")
        rob_acc.append(robust_accuracy.cpu().float())
        if batch_idx % 10 == 0:
            # 可视化整10个batch的生成质量
            adv_target = torch.max(fmodel(raw_advs),1)[1]
            plot_clean_and_adver(adver_example = raw_advs, adver_target = adv_target, clean_example= images,
                                 clean_target =labels, attack_name= attack_method, dataset = dataset)
        # 数据的保存
        # generate_imgs(raw_advs, adv_png_dataroot)
        for j, img in enumerate(raw_advs):
            img_to_save = img.clone()
            filename = adv_png_dataroot + "img_" + str(batch*batch_idx+j) + ".png"
            torchvision.utils.save_image(img_to_save, filename, normalize=True)
        save_adv_image.append(raw_advs.cpu())
        save_label.append(labels.cpu())
    save_adv_image = np.concatenate(save_adv_image, axis=0)
    save_label = np.concatenate(save_label, axis=0)
    np.save('{}image'.format(np_dataroot), save_adv_image)
    np.save('{}label'.format(np_dataroot), save_label)
    print('Grad data saved.')
    # 结束生成和保存
    print("------------final-----------")
    print(f"clean accuracy:  {np.average(acc) * 100:.1f} %")
    print(f"robust accuracy:  {np.average(rob_acc) * 100:.1f} %")

def quality_IS(dataset, np_data_root):
    save_adversaries = np.load('{}image.npy'.format((np_data_root)), allow_pickle= True)
    if dataset == 'mnist' or dataset == 'fashion':
        preprocessed_advs = preprocess_1D_imgs(save_adversaries)
        mean_is, std_is = inception_score(preprocessed_advs, cuda=True, batch_size=32, resize=True, splits=1)
    elif dataset == 'cifar10' or dataset == 'svhn':
        # print(save_adversaries.shape)
        mean_is, std_is = inception_score(save_adversaries, cuda=True, batch_size=32, resize=True, splits=1)
    print('inception_score:', mean_is)
    return mean_is

def quality_FID(batch, clean_data_root, adv_data_root):
    # fid_score_64 = calculate_fid_given_paths([clean_data_root, adv_data_root], batch_size=batch, device=0, dims=64)
    # print('fid_score_64:', fid_score_64)
    fid_score_2048 = calculate_fid_given_paths([clean_data_root, adv_data_root], batch_size=batch, device=0, dims=2048)
    print('fid_score_2048:', fid_score_2048)
    return fid_score_2048

def gentle_filter(loader, train_loader, model, dataset, cluster_num, sample_mean, precision, select_num):
    feature_list, num_output = metrics.get_information(model, dataset)
    gentle_score = lib.get_gentle_score_without_label(model, loader, cluster_num, "OOD", sample_mean, precision, num_output - 1)

    M_in_train = lib.get_gentle_score(model, train_loader, cluster_num, True, sample_mean, precision, num_output - 1)
    M_in_train = np.asarray(M_in_train, dtype=np.float32)
    Mahalanobis_in_train = M_in_train.reshape((M_in_train.shape[0], -1))
    Mahalanobis_in_train = np.asarray(Mahalanobis_in_train, dtype=np.float32)
    Mahalanobis_in_train = np.array(Mahalanobis_in_train).flatten()  # score(T_train)
    dis1 = []
    for i in range(len(gentle_score)):
        tmp_score = np.asarray(gentle_score[i])
        tmp_score = np.array(tmp_score).flatten()
        tmp_dis = wasserstein_distance(Mahalanobis_in_train, tmp_score)
        dis1.append(tmp_dis)
    idx_list = np.argsort(dis1)  # dis1 从小到大排序对应的 idx
    select = idx_list[:select_num]
    return select

def init_nac(dataset, model):
    act_id = []
    model.eval()

    if dataset == 'mnist' or dataset == 'fashion':
        temp_x = torch.rand(1, 1, 28, 28).cuda()
    elif dataset == 'cifar10' or dataset == 'svhn':
        temp_x = torch.rand(1, 3, 32, 32).cuda()
    else:
        print("Don't know the shape of input dataset.")
        temp_x = None
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    for i in range(num_output):
        neurons_num = temp_list[i].size(1)
        act_id.append(torch.zeros(neurons_num))
    return act_id

def gan_generation(bad_gan_model_path, good_gan_model_path, dataset, batch, itertime, ganroot):
    if dataset == 'mnist' or dataset == 'fashion':
        # # GAN initial
        # latent_dim = 100
        # img_shape = (1, 28, 28)
        # bad_gannet = torch.load(bad_gan_model_path, map_location="cuda:0")
        # good_gannet = torch.load(good_gan_model_path, map_location="cuda:0")
        # generator_bad_gan = Generator_gan(img_shape=img_shape, latent_dim=latent_dim).cuda()
        # generator_good_gan = Generator_gan(img_shape=img_shape, latent_dim=latent_dim).cuda()
        # generator_bad_gan.load_state_dict(bad_gannet['generator'])
        # generator_good_gan.load_state_dict(good_gannet['generator'])
        # generator_bad_gan.cuda()
        # generator_bad_gan.eval()
        # generator_good_gan.cuda()
        # generator_good_gan.eval()
        # DCGAN initial
        latent_dim = 100
        img_shape = (1, 28, 28)
        bad_gannet = torch.load(bad_gan_model_path, map_location="cuda:0")
        generator_bad_gan = Generator_gan(img_shape=img_shape, latent_dim=latent_dim).cuda()
        # bad_gannet = torch.load(bad_gan_model_path, map_location="cuda:0")
        # generator_bad_gan = Generator_dcgan(latent_dim=latent_dim, channels=1).cuda()
        good_gannet = torch.load(good_gan_model_path, map_location="cuda:0")
        generator_good_gan = Generator_dcgan(latent_dim=latent_dim, channels=1).cuda()
        generator_bad_gan.load_state_dict(bad_gannet['generator'])
        generator_good_gan.load_state_dict(good_gannet['generator'])
        generator_bad_gan.cuda()
        generator_bad_gan.eval()
        generator_good_gan.cuda()
        generator_good_gan.eval()
    elif dataset == 'cifar10' or dataset == 'svhn':
        latent_dim = 100
        img_shape = (3, 32, 32)
        bad_gannet = torch.load(bad_gan_model_path, map_location="cuda:0")
        good_gannet = torch.load(good_gan_model_path, map_location="cuda:0")
        generator_bad_gan = Generator_dcgan(latent_dim=latent_dim, channels=3).cuda()
        generator_good_gan = Generator_dcgan(latent_dim=latent_dim, channels=3).cuda()
        generator_bad_gan.load_state_dict(bad_gannet['generator'])
        generator_good_gan.load_state_dict(good_gannet['generator'])
        generator_bad_gan.cuda()
        generator_bad_gan.eval()
        generator_good_gan.cuda()
        generator_good_gan.eval()

    # 基于GAN生成图像
    os.makedirs(ganroot, exist_ok=True)
    gan_png_dataroot = os.path.expanduser(os.path.join(ganroot) + 'gan-png/')
    np_dataroot = os.path.expanduser(os.path.join(ganroot) + 'gan-np/')
    os.makedirs(gan_png_dataroot, exist_ok=True)
    os.makedirs(np_dataroot, exist_ok=True)
    save_gan_image = []
    for i in range(itertime):
        noises = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batch, latent_dim))))
        bad_fake_images = generator_bad_gan(noises)
        good_fake_images = generator_good_gan(noises)
        if dataset == 'mnist' or dataset == 'fashion':
            resize = transforms.Compose([transforms.Resize(28)])
            good_fake_images = resize(good_fake_images)
        for j, img in enumerate(bad_fake_images):
            img_to_save = img.clone()
            filename = gan_png_dataroot + "bad_img_" + str(batch*i+j) + ".png"
            torchvision.utils.save_image(img_to_save, filename, normalize=True)
        save_gan_image.append(bad_fake_images.detach().cpu())
        for j, img in enumerate(good_fake_images):
            img_to_save = img.clone()
            filename = gan_png_dataroot + "good_img_" + str(batch*i+j) + ".png"
            torchvision.utils.save_image(img_to_save, filename, normalize=True)
        save_gan_image.append(good_fake_images.detach().cpu())
    save_gan_image = np.concatenate(save_gan_image, axis=0)
    np.save('{}image'.format(np_dataroot), save_gan_image)
    print('GAN data saved.')

def save_clean_png(dataloader, dataset, batch):
    clean_png_root = './data/' + dataset + '-data' + '/png/'
    if os.path.exists(clean_png_root): # 如果已经保存过png，则直接返回
        return
    os.makedirs(clean_png_root, exist_ok=True)
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.cuda(), labels.cuda()
        for j, img in enumerate(images):
            img_to_save = img.clone()
            filename = clean_png_root + "img_" + str(batch*batch_idx+j) + ".png"
            torchvision.utils.save_image(img_to_save, filename, normalize=True)
    print('clean png data saved.')

def trans_generation(transroot, dataset, batch):
    trans_png_dataroot = os.path.expanduser(os.path.join(transroot) + 'trans-png/')
    trans_np_dataroot = os.path.expanduser(os.path.join(transroot) + 'trans-np/')
    # if os.path.exists(trans_png_dataroot) and os.path.exists(trans_np_dataroot): # 如果已经保存过数据，则直接返回
    #     return
    os.makedirs(trans_png_dataroot, exist_ok=True)
    os.makedirs(trans_np_dataroot, exist_ok=True)
    data_root = os.path.expanduser(os.path.join(dataroot, str(dataset) + '-data'))
    if dataset == 'mnist':
        small_aug = datasets.MNIST(root=data_root, train=False, download=True, transform=mnist_small_trans)
        large_aug = datasets.MNIST(root=data_root, train=False, download=True, transform=mnist_large_trans)
    elif dataset == 'fashion':
        small_aug = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=fashion_small_trans)
        large_aug = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=fashion_large_trans)
    elif dataset == 'cifar10':
        small_aug = datasets.CIFAR10(root=data_root, train=False, download=True, transform=cifar_small_trans)
        large_aug = datasets.CIFAR10(root=data_root, train=False, download=True, transform=cifar_large_trans)
    elif dataset == 'svhn':
        small_aug = datasets.SVHN(root=data_root, split='test', download=True, transform=svhn_small_trans)
        large_aug = datasets.SVHN(root=data_root, split='test', download=True, transform=svhn_large_trans)
    data = small_aug + large_aug
    loader = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=random, num_workers=1)
    images = []
    for batch_idx, (image,_) in enumerate(loader):
        images.append(image)
        for j,img in enumerate(image):
            img_to_save = img.clone()
            filename = trans_png_dataroot + 'img_' + str(batch_idx*batch+j) + '.png'
            torchvision.utils.save_image(img_to_save, filename, normalize=True)
        if batch_idx >=19:
            break
    images = np.concatenate(images, axis=0)
    np.save('{}image'.format(trans_np_dataroot), images)
    print('Trans data saved.')

def data_reload(saveroot, dataset, type, batch, sampler=None):
    if type == 'grad':
        np_dataroot = os.path.expanduser(os.path.join(saveroot) + 'adv-np/')
        save_data = np.load('{}image.npy'.format((np_dataroot)), allow_pickle=True)
    elif type == 'gan':
        np_dataroot = os.path.expanduser(os.path.join(saveroot) + 'gan-np/')
        save_data = np.load('{}image.npy'.format((np_dataroot)), allow_pickle=True)
    elif type == 'trans':
        np_dataroot = os.path.expanduser(os.path.join(saveroot) + 'trans-np/')
        save_data = np.load('{}image.npy'.format((np_dataroot)), allow_pickle=True)
    if dataset == 'mnist':
        tmp_data = data_utils.MNIST_local_reload_without_label(root=data_root, data=save_data)
        loader = torch.utils.data.DataLoader(tmp_data, batch_size=batch, shuffle=False, num_workers=0, sampler=sampler)
    elif dataset == 'fashion':
        tmp_data = data_utils.Fashion_local_reload_without_label(root=data_root, data=save_data)
        loader = torch.utils.data.DataLoader(tmp_data, batch_size=batch, shuffle=False, num_workers=0, sampler=sampler)
    elif dataset == 'cifar10':
        tmp_data = data_utils.CIFAR10_local_reload_without_label(root=data_root, data=save_data)
        loader = torch.utils.data.DataLoader(tmp_data, batch_size=batch, shuffle=False, num_workers=0, sampler=sampler)
    elif dataset == 'svhn':
        tmp_data = data_utils.SVHN_local_reload_without_label(root=data_root, data=save_data)
        loader = torch.utils.data.DataLoader(tmp_data, batch_size=batch, shuffle=False, num_workers=0, sampler=sampler)
    return loader

def quality_evaluation(dataroot, dataset, batch, generation_type, filter, filter_type):
    filter_data_loader = data_reload(dataroot, dataset, generation_type, batch, filter)
    filter_png_dataroot = os.path.expanduser(os.path.join(dataroot)+filter_type+'-png/')
    filter_np_dataroot = os.path.expanduser(os.path.join(dataroot)+filter_type+'-np/')
    os.makedirs(filter_png_dataroot, exist_ok=True)
    os.makedirs(filter_np_dataroot, exist_ok=True)
    images = []
    for batch_idx, image in enumerate(filter_data_loader):
        images.append(image.cpu())
        for j, img in enumerate(image):
            img_to_save = img.clone()
            filename = filter_png_dataroot + 'img_' + str(batch_idx * batch + j) + '.png'
            torchvision.utils.save_image(img_to_save, filename, normalize=True)
    images = np.concatenate(images, axis=0)
    np.save('{}image'.format(filter_np_dataroot), images)
    print(f"------------{filter_type} quality-----------")
    clean_png_root = './data/' + dataset + '-data' + '/png/'
    filter_fid = quality_FID(batch, clean_png_root, filter_png_dataroot)
    filter_is = quality_IS(dataset, filter_np_dataroot)
    return filter_fid, filter_is

def nac_filter(loader, batch, model, dataset, select_num, nc_k):
    # 初始化NC覆盖矩阵
    nc_act_id = init_nac(dataset, model)
    select = []
    for batch_idx, (images) in enumerate(loader):
        for j, img in enumerate(images):
            idx = batch_idx * batch +j
            nc_act_id, select_flag = calc_nac_increase(model, img, nc_k, nc_act_id)
            if select_flag == True and len(select) < select_num:
                select.append(idx)
            elif len(select) == select_num:
                return select
        print('current batch', batch_idx, 'current selected', len(select))
    while len(select) < select_num:
        idx = random.randint(0, len(data_loader.dataset)-1)
        if idx not in select:
            select.append(idx)
    return select

def snac_filter(loader, batch, model, dataset, select_num, upper):
    # 初始化NC覆盖矩阵
    nc_act_id = init_nac(dataset, model)
    select = []
    for batch_idx, (images) in enumerate(loader):
        for j, img in enumerate(images):
            idx = batch_idx * batch +j
            nc_act_id, select_flag = calc_snac_increase(model, img, upper, nc_act_id)
            if select_flag == True and len(select) < select_num:
                select.append(idx)
            elif len(select) == select_num:
                return select
        print('current batch', batch_idx, 'current selected', len(select))
    while len(select) < select_num:
        idx = random.randint(0, len(data_loader.dataset) - 1)
        if idx not in select:
            select.append(idx)
    return select

def tknc_filter(loader, batch, model, dataset, select_num):
    # 初始化NC覆盖矩阵
    nc_act_id = init_nac(dataset, model)
    select = []
    for batch_idx, (images) in enumerate(loader):
        for j, img in enumerate(images):
            idx = batch_idx * batch +j
            nc_act_id, select_flag = calc_tknc_increase(model, img, nc_act_id)
            if select_flag == True and len(select) < select_num:
                select.append(idx)
            elif len(select) == select_num:
                return select
        print('current batch', batch_idx, 'current selected', len(select))
    while len(select) < select_num:
        idx = random.randint(0, len(data_loader.dataset) - 1)
        if idx not in select:
            select.append(idx)
    return select

def nbc_filter(loader, batch, model, dataset, select_num, upper, lower):
    # 初始化NC覆盖矩阵
    nc_act_id_upper = init_nac(dataset, model)
    nc_act_id_lower = init_nac(dataset, model)
    select = []
    for batch_idx, (images) in enumerate(loader):
        for j, img in enumerate(images):
            idx = batch_idx * batch +j
            nac_at_id_upper, nc_act_id_lower, select_flag = calc_nbc_increase(model, img, upper, lower,
                                                                              nc_act_id_upper, nc_act_id_lower)
            if select_flag == True and len(select) < select_num:
                select.append(idx)
            elif len(select) == select_num:
                return select
        print('current batch', batch_idx, 'current selected', len(select))
    while len(select) < select_num:
        idx = random.randint(0, len(data_loader.dataset) - 1)
        if idx not in select:
            select.append(idx)
    return select

def calc_nac_increase(model, data, nc_k, act_id):
    new_act_id = act_id
    select_flag = False
    model.eval()

    with torch.no_grad():
        data = data.unsqueeze(0).cuda()
        data = Variable(data)
        out_features = model.feature_list(data)[1]
    layer_num = len(out_features)
    for i in range(layer_num): # 第i层的输出
        neurons_num = out_features[i].size(1)
        tmp_test = out_features[i]
        for k in range(neurons_num):
            tmp = tmp_test[:,k]
            if tmp > nc_k and act_id[i][k] == 0:
                new_act_id[i][k] = 1
                select_flag = True

    return new_act_id, select_flag

def calc_snac_increase(model, data, upper, act_id):
    new_act_id = act_id
    select_flag = False
    model.eval()

    with torch.no_grad():
        data = data.unsqueeze(0).cuda()
        data = Variable(data)
        out_features = model.feature_list(data)[1]
    layer_num = len(out_features)
    for i in range(layer_num): # 第i层的输出
        neurons_num = out_features[i].size(1)
        tmp_test = out_features[i]
        for k in range(neurons_num):
            tmp = tmp_test[:,k]
            if tmp > upper[i][k] and act_id[i][k] == 0:
                new_act_id[i][k] = 1
                select_flag = True
    return new_act_id, select_flag

def calc_nbc_increase(model, data, upper, lower, act_id_upper, act_id_lower):
    new_act_id_upper = act_id_upper
    new_act_id_lower = act_id_lower
    select_flag = False
    model.eval()

    with torch.no_grad():
        data = data.unsqueeze(0).cuda()
        data = Variable(data)
        out_features = model.feature_list(data)[1]
    layer_num = len(out_features)
    for i in range(layer_num): # 第i层的输出
        neurons_num = out_features[i].size(1)
        tmp_test = out_features[i]
        for k in range(neurons_num):
            tmp = tmp_test[:,k]
            if tmp >= upper[i][k] and act_id_upper[i][k] == 0:
                new_act_id_upper[i][k] = 1
                select_flag = True
            if tmp <= lower[i][k] and act_id_lower[i][k] == 0:
                new_act_id_lower[i][k] = 1
                select_flag = True
    return new_act_id_upper, new_act_id_lower, select_flag

def calc_tknc_increase(model, data, act_id):
    new_act_id = act_id
    select_flag = False
    model.eval()

    with torch.no_grad():
        data = data.unsqueeze(0).cuda()
        data = Variable(data)
        out_features = model.feature_list(data)[1]
    layer_num = len(out_features)
    for i in range(layer_num): # 第i层的输出
        tmp_test = out_features[i]
        val, index = torch.max(tmp_test, 1)
        if act_id[i][index] == 0:
            new_act_id[i][index] = 1
            select_flag = True
    return new_act_id, select_flag

def lsc_filter(data_loader, model, dataset, select_num, kdes, removed_cols):
    target_ats, target_pred = get_ats_without_label(model, data_loader)
    lsa, select = [], []
    for i, at in enumerate(target_ats):
        label = target_pred[i]
        kde = kdes[label]
        refined_at = np.delete(at, removed_cols, axis=0)
        tmp_lsa = np.asscalar(-kde.logpdf(np.transpose(refined_at)))
        lsa.append(tmp_lsa)
    idx_list = np.argsort(lsa)[::-1]
    if dataset == 'mnist' or dataset == 'fashion':
        ub, n = 2000, 1000
    elif dataset == 'cifar10' or dataset == 'svhn':
        ub, n = 100, 1000
    bucket_l = ub/n
    covered_lsc = [0] * n
    for i in range(n):
        lower = bucket_l * i
        upper = bucket_l * (i+1)
        for j in range(len(lsa)):
            if lsa[j] > lower and lsa[j] <= upper:
                covered_lsc[i] = 1
                select.append(j)
                if len(select) >= select_num:
                    return select
                break
    while len(select) < select_num:
        j = random.randint(0, len(lsa)-1)
        if j not in select:
            select.append(j)
            if len(select) >= select_num:
                return select

def nc_filter(loader, batch, model, dataset, select_num, nc_k, upper, lower):
    nac_act_id = init_nac(dataset, model)
    snac_act_id, tknc_act_id, nbc_act_id_lower, nbc_act_id_upper = nac_act_id, nac_act_id, nac_act_id, nac_act_id
    nac_select, snac_select, nbc_select, tknc_select = [], [], [], []
    for batch_idx, (images) in enumerate(loader):
        for j, img in enumerate(images):
            idx = batch_idx * batch +j
            nac_act_id, snac_act_id, tknc_act_id, nbc_act_id_lower, nbc_act_id_upper, nac_select_flag, snac_select_flag, tknc_select_flag, nbc_select_flag = \
                calc_nc_increase(model, img, nc_k, upper, lower, nac_act_id, snac_act_id, tknc_act_id, nbc_act_id_lower, nbc_act_id_upper)
            # nac_act_id, nac_select_flag = calc_nac_increase(model, img, nc_k, nac_act_id)
            # snac_act_id, snac_select_flag = calc_snac_increase(model, img, upper, snac_act_id)
            # tknc_act_id, tknc_select_flag = calc_tknc_increase(model, img, tknc_act_id)
            # nbc_at_id_upper, nbc_act_id_lower, nbc_select_flag = calc_nbc_increase(model, img, upper, lower,
            #                                                                   nbc_act_id_upper, nbc_act_id_lower)
            if len(nac_select) < select_num and nac_select_flag == True:
                nac_select.append(idx)
            if len(snac_select) < select_num and snac_select_flag == True:
                snac_select.append(idx)
            if len(nbc_select) < select_num and nbc_select_flag == True:
                nbc_select.append(idx)
            if len(tknc_select) < select_num and tknc_select_flag == True:
                tknc_select.append(idx)
            if len(nac_select) == select_num and len(snac_select) == select_num and len(nbc_select) == select_num and len(tknc_select) == select_num:
                return nac_select, snac_select, nbc_select, tknc_select
        print('Current batch:', batch_idx, '    NAC:', len(nac_select), '    SNAC:', len(snac_select),
              '    NBC:', len(nbc_select), '    TKNC:', len(tknc_select))
    for select in [nac_select, snac_select, nbc_select, tknc_select]:
        while len(select) < select_num:
            idx = random.randint(0, len(data_loader.dataset) - 1)
            if idx not in select:
                select.append(idx)
    return nac_select, nbc_select, snac_select, tknc_select

def calc_nc_increase(model, data, nc_k, upper, lower, nac_act_id, snac_act_id, tknc_act_id, nbc_act_id_lower,
                     nbc_act_id_upper):
    new_nac_act_id = nac_act_id
    new_snac_act_id = snac_act_id
    new_tknc_act_id = tknc_act_id
    new_nbc_act_id_lower = nbc_act_id_lower
    new_nbc_act_id_upper = nbc_act_id_upper
    nac_select_flag, snac_select_flag, tknc_select_flag, nbc_select_flag = False, False, False, False
    model.eval()

    with torch.no_grad():
        data = data.unsqueeze(0).cuda()
        data = Variable(data)
        out_features = model.feature_list(data)[1]
    layer_num = len(out_features)
    for i in range(layer_num): # 第i层的输出
        neurons_num = out_features[i].size(1)
        tmp_test = out_features[i]
        for k in range(neurons_num):
            tmp = tmp_test[:,k]
            if tmp > nc_k and nac_act_id[i][k] == 0:
                new_nac_act_id[i][k] = 1
                nac_select_flag = True
            if tmp > upper[i][k] and snac_act_id[i][k] == 0:
                new_snac_act_id[i][k] = 1
                snac_select_flag = True
            if tmp >= upper[i][k] and nbc_act_id_upper[i][k] == 0:
                new_nbc_act_id_upper[i][k] = 1
                nbc_select_flag = True
            if tmp <= lower[i][k] and nbc_act_id_lower[i][k] == 0:
                new_nbc_act_id_lower[i][k] = 1
                nbc_select_flag = True
            val, index = torch.max(tmp_test, 1)
            if tknc_act_id[i][index] == 0:
                new_tknc_act_id[i][index] = 1
                tknc_select_flag = True
    return new_nac_act_id, new_snac_act_id, new_tknc_act_id, new_nbc_act_id_lower, new_nbc_act_id_upper, nac_select_flag, snac_select_flag, tknc_select_flag, nbc_select_flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch code: Train the models.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataset', required=True, help='cifar10 | svhn | mnist | fashion')
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--generation_type', default='grad', type=str, help=('grad | gan | trans'))
    parser.add_argument('--pre_trained_model', default='./pre_trained', help='path to pre_trained_models')
    parser.add_argument('--net_type', required=True, help='resnet18 | vgg16 | densenet | vgg16')
    parser.add_argument('--outf', default='./output/', help='folder to output results')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--Grad_flag', type=int, default=1, help='run the gradient-guided exp')
    parser.add_argument('--GAN_flag', type=int, default=1, help='run the gradient-guided exp')
    parser.add_argument('--Trans_flag', type=int, default=1, help='run the gradient-guided exp')
    args = parser.parse_args()
    print(args)

    os.makedirs('./print/', exist_ok=True)


    now = datetime.now()
    strtime = now.strftime('%b%d%H%M')
    outfile = './print/' + args.dataset + '_' + args.net_type + '_rq3_'+ strtime +'.txt'
    sys.stdout = Logger(outfile)

    Grad_flag, GAN_flag, Trans_flag = args.Grad_flag, args.GAN_flag, args.Trans_flag
    dataset, net_type, dataroot = args.dataset, args.net_type, args.dataroot
    batch = args.batch_size

    pre_trained_dir = args.pre_trained_model + "/" + args.net_type + "_" + args.dataset + "/"
    pre_trained_net = pre_trained_dir + "/" + args.net_type + "_" + args.dataset + "_aug0.pth"
    outf = './evaluation_rq3/' + dataset + '_' + net_type + '/'
    os.makedirs(outf, exist_ok=True)

    # 确定并载入模型
    print('==> Preparing model..')
    num_classes, cluster_num = 10, 10
    if dataset == 'cifar10' or dataset == 'svhn':
        if net_type == "densenet":
            model = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
            re_model = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
        elif net_type == "vgg16":
            model = cifar_VGG('VGG16').to("cuda")
            re_model = cifar_VGG('VGG16').to("cuda")
    if dataset == 'fashion' or dataset == 'mnist':
        if net_type == "resnet18":
            model = mnist_ResNet18(num_c=num_classes).to("cuda")
            re_model = mnist_ResNet18(num_c=num_classes).to("cuda")
        elif net_type == "vgg11":
            model = mnist_VGG('VGG11').to("cuda")
            re_model = mnist_VGG('VGG11').to("cuda")
    model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu))['net'])
    model.cuda().eval()

    # 干净数据集的png准备
    print('==> Saving the clean png images..')
    data_root = os.path.expanduser(os.path.join("./data", str(dataset) + '-data'))
    train_loader, valid_loader = get_dataloader(dataset, data_root, batch)
    save_clean_png(valid_loader, dataset, batch)
    clean_png_root = './data/' + dataset + '-data' + '/png/'

    #########################################################
    #######         Grad-based test generation        #######
    #########################################################
    if Grad_flag == 1:
        # 载入数据集和模型用于对抗生成
        print('==> Gradient-based generation <==')
        generation_type = 'grad'
        grad_train_loader, grad_valid_loader, bounds, preprocessing = get_grad_dataloader(dataset, data_root, batch)

        fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
        gradroot = os.path.expanduser(os.path.join(outf) + 'grad/')
        os.makedirs(gradroot, exist_ok=True)
        ### waiting to add: 不同攻击方法的测试生成与评估
        # attack_mothods = ['PGD', 'FGSM', 'CW', 'deepfool']
        attack_method = 'FGSM'
        grad_attack = fb.attacks.FGSM()
        # attack_method = 'PGD'
        # grad_attack = fb.attacks.PGD()
        grad_generation(grad_valid_loader, fmodel, grad_attack, gradroot, attack_method, dataset)
        # 整体的质量报告
        # print("------------overall quality-----------")
        # adv_png_dataroot = os.path.expanduser(os.path.join(gradroot) + 'adv-png/')
        # grad_np_dataroot = os.path.expanduser(os.path.join(gradroot) + 'adv-np/')
        # quality_FID(batch, clean_png_root, adv_png_dataroot)
        # quality_IS(dataset, grad_np_dataroot)

        print("==> Method-Guided generation simulation")
        select_num = 500
        generation_type = 'grad'
        data_loader = data_reload(gradroot, dataset, generation_type, batch)
        print('Total num:', len(data_loader.dataset), 'Select target num:', select_num)
        train_loader, _ = get_dataloader(dataset, data_root, batch)
        # 准备gentle所需信息
        cluster_root = './output/' + dataset + '_' + net_type + '/'
        mean_path = cluster_root + str(cluster_num) + 'classes_mean.npy'
        precision_path = cluster_root + str(cluster_num) + 'classes_precision.npy'
        sample_mean, precision = selection.get_mean_precision(model, net_type, dataset, dataroot, cluster_num,
                                                              batch, train_loader, outf, mean_path, precision_path)

        print("===> Overall-guided")
        str_time = time.time()
        filter_type = 'overall'
        overall_filter_idx = []
        while len(overall_filter_idx) < select_num:
            idx = random.randint(0, len(data_loader.dataset)-1)
            if idx not in overall_filter_idx:
                overall_filter_idx.append(idx)
        print(filter_type, len(overall_filter_idx))
        # 评估loader出数据的质量
        quality_evaluation(gradroot, dataset, batch, generation_type, overall_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))

        print("===> Gentle-guided")
        str_time = time.time()
        filter_type = 'gentle'
        gentle_filter_idx = gentle_filter(data_loader, train_loader, model, dataset, cluster_num, sample_mean, precision,
                                          select_num)
        print(filter_type, len(gentle_filter_idx))
        # 评估loader出数据的质量
        quality_evaluation(gradroot, dataset, batch, generation_type, gentle_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))

        print("===> Neuron coverage-guided")
        str_time = time.time()
        # 准备 NBC 和 SNAC 的数据
        upper, lower, neuron_num = metrics.get_upper_lower(model, train_loader, dataset)
        nc_k = 0.75
        print('neuron number:', neuron_num)
        print('model shape:', len(upper))
        for i in upper:
            print('layer:', len(i))

        nac_filter_idx, nbc_filter_idx, snac_filter_idx, tknc_filter_idx = nc_filter(data_loader, batch, model, dataset, select_num, nc_k, upper, lower)
        filter_type = 'nac'
        print('nac_filter_idx', len(nac_filter_idx))
        quality_evaluation(gradroot, dataset, batch, generation_type, nac_filter_idx, filter_type)
        filter_type = 'nbc'
        print('nbc_filter_idx', len(nbc_filter_idx))
        quality_evaluation(gradroot, dataset, batch, generation_type, nbc_filter_idx, filter_type)
        filter_type = 'snac'
        print('snac_filter_idx', len(snac_filter_idx))
        quality_evaluation(gradroot, dataset, batch, generation_type, snac_filter_idx, filter_type)
        filter_type = 'tknc'
        print('tknc_filter_idx', len(tknc_filter_idx))
        quality_evaluation(gradroot, dataset, batch, generation_type, tknc_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))

        print("==> LSC-guided")
        str_time = time.time()
        filter_type = 'lsc'
        lsa_threshold = 0.01
        lsc_train_loader = train_loader
        lsc_data_loader = data_loader
        if dataset == 'mnist' and net_type == 'resnet18':
            lsa_threshold = 0.08
        elif dataset == 'mnist' and net_type == 'vgg11':
            lsa_threshold = 0.008
        elif dataset == 'fashion' and net_type == 'vgg11':
            lsa_threshold = 0.01
        elif dataset == 'fashion' and net_type == 'resnet18':
            lsa_threshold = 0.15
        elif dataset == 'svhn' and net_type == 'vgg16':
            lsa_threshold = 0.003
        elif dataset == 'svhn' and net_type == 'densenet':
            lsa_threshold = 0.03
            batch_lsc = 64
            lsc_train_loader, _ = get_dataloader(dataset, data_root, batch_lsc)
            lsc_data_loader = data_reload(gradroot, dataset, generation_type, batch_lsc)
        elif dataset == 'cifar10' and net_type == 'densenet':
            lsa_threshold = 0.01
            batch_lsc = 64
            lsc_train_loader, _ = get_dataloader(dataset, data_root, batch_lsc)
            lsc_data_loader = data_reload(gradroot, dataset, generation_type, batch_lsc)
        elif dataset == 'cifar10' and net_type == 'vgg16':
            lsa_threshold = 0.00002
        kdes, removed_cols = get_lsa_kdes(model, lsc_train_loader, lsa_threshold=lsa_threshold)
        lsc_filter_idx = lsc_filter(lsc_data_loader, model, dataset, select_num, kdes, removed_cols)
        print('lsc_filter_idx', len(lsc_filter_idx))
        quality_evaluation(gradroot, dataset, batch, generation_type, lsc_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))


    #########################################################
    #######         GAN-based test generation         #######
    #########################################################
    if GAN_flag == 1:

        print("==> GAN-based generation <==")
        generation_type = 'gan'
        ganroot = os.path.expanduser(os.path.join(outf) + 'gan/')
        itertime = 10 # 基于batch迭代生成的次数
        if dataset == 'mnist' or dataset == 'fashion':
            bad_gan_model_path = "pre_trained/" + dataset + "/gan-bad.pth"
            # good_gan_model_path = "pre_trained/" + dataset + "/gan-good.pth"
            # bad_gan_model_path = "pre_trained/" + dataset + "/dcgan-50epoch.pth"
            good_gan_model_path = "pre_trained/" + dataset + "/dcgan-200epoch.pth"
        elif dataset == 'svhn':
            bad_gan_model_path = "pre_trained/" + dataset + "/dcgan-100epoch.pth"
            good_gan_model_path = "pre_trained/" + dataset + "/dcgan-400epoch.pth"
        gan_generation(bad_gan_model_path, good_gan_model_path, dataset, batch, itertime, ganroot)

        # # 整体的质量报告
        # print("------------overall quality-----------")
        # gan_png_dataroot = os.path.expanduser(os.path.join(ganroot) + 'gan-png/')
        # gan_np_dataroot = os.path.expanduser(os.path.join(ganroot) + 'gan-np/')
        # quality_FID(batch, clean_png_root, gan_png_dataroot)
        # quality_IS(dataset, gan_np_dataroot)

        print("==> Method-Guided generation simulation")
        select_num = 500
        generation_type = 'gan'
        data_loader = data_reload(ganroot, dataset, generation_type, batch)
        print('Total num:', len(data_loader.dataset), 'Select target num:', 500)
        train_loader, _= get_dataloader(dataset, data_root, batch)
        # 准备gentle所需信息
        cluster_root = './output/' + dataset + '_' + net_type + '/'
        mean_path = cluster_root + str(cluster_num) + 'classes_mean.npy'
        precision_path = cluster_root + str(cluster_num) + 'classes_precision.npy'
        sample_mean, precision = selection.get_mean_precision(model, net_type, dataset, dataroot, cluster_num,
                                                              batch, train_loader, outf, mean_path, precision_path)

        print("===> Overall-guided")
        str_time = time.time()
        filter_type = 'overall'
        overall_filter_idx = []
        while len(overall_filter_idx) < select_num:
            idx = random.randint(0, len(data_loader.dataset)-1)
            if idx not in overall_filter_idx:
                overall_filter_idx.append(idx)
        print(filter_type, len(overall_filter_idx))
        # 评估loader出数据的质量
        quality_evaluation(ganroot, dataset, batch, generation_type, overall_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))

        print("===> Gentle-guided")
        str_time = time.time()
        filter_type = 'gentle'
        gentle_filter_idx = gentle_filter(data_loader, train_loader, model, dataset, cluster_num, sample_mean, precision,
                                              select_num)
        print(filter_type, len(gentle_filter_idx))
        # 评估loader出数据的质量
        quality_evaluation(ganroot, dataset, batch, generation_type, gentle_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))

        print("===> Neuron coverage-guided")
        str_time = time.time()
        # 准备 NBC 和 SNAC 的数据
        upper, lower, neuron_num = metrics.get_upper_lower(model, train_loader, dataset)
        nc_k = 0.75
        print('neuron number', neuron_num)
        print('model shape', len(upper))
        for i in upper:
            print('layer:', len(i))

        nac_filter_idx, nbc_filter_idx, snac_filter_idx, tknc_filter_idx = nc_filter(data_loader, batch, model, dataset,
                                                                                     select_num, nc_k, upper, lower)
        filter_type = 'nac'
        print('nac_filter_idx', len(nac_filter_idx))
        quality_evaluation(ganroot, dataset, batch, generation_type, nac_filter_idx, filter_type)
        filter_type = 'nbc'
        print('nbc_filter_idx', len(nbc_filter_idx))
        quality_evaluation(ganroot, dataset, batch, generation_type, nbc_filter_idx, filter_type)
        filter_type = 'snac'
        print('snac_filter_idx', len(snac_filter_idx))
        quality_evaluation(ganroot, dataset, batch, generation_type, snac_filter_idx, filter_type)
        filter_type = 'tknc'
        print('tknc_filter_idx', len(tknc_filter_idx))
        quality_evaluation(ganroot, dataset, batch, generation_type, tknc_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))



        print("==> LSC-guided")
        str_time = time.time()
        filter_type = 'lsc'
        lsa_threshold = 0.01
        lsc_train_loader = train_loader
        lsc_data_loader = data_loader
        if dataset == 'mnist' and net_type == 'resnet18':
            lsa_threshold = 0.08
        elif dataset == 'mnist' and net_type == 'vgg11':
            lsa_threshold = 0.008
        elif dataset == 'fashion' and net_type == 'vgg11':
            lsa_threshold = 0.01
        elif dataset == 'fashion' and net_type == 'resnet18':
            lsa_threshold = 0.15
        elif dataset == 'svhn' and net_type == 'vgg16':
            lsa_threshold = 0.003
        elif dataset == 'svhn' and net_type == 'densenet':
            lsa_threshold = 0.03
            batch_lsc = 64
            lsc_train_loader, _ = get_dataloader(dataset, data_root, batch_lsc)
            lsc_data_loader = data_reload(ganroot, dataset, generation_type, batch_lsc)
        elif dataset == 'cifar10' and net_type == 'densenet':
            lsa_threshold = 0.01
            batch_lsc = 64
            lsc_train_loader, _ = get_dataloader(dataset, data_root, batch_lsc)
            lsc_data_loader = data_reload(ganroot, dataset, generation_type, batch_lsc)
        elif dataset == 'cifar10' and net_type == 'vgg16':
            lsa_threshold = 0.00002
        kdes, removed_cols = get_lsa_kdes(model, lsc_train_loader, lsa_threshold=lsa_threshold)
        lsc_filter_idx = lsc_filter(lsc_data_loader, model, dataset, select_num, kdes, removed_cols)
        print('lsc_filter_idx', len(lsc_filter_idx))
        quality_evaluation(ganroot, dataset, batch, generation_type, lsc_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))

    #########################################################
    #######        Trans-based test generation        #######
    #########################################################

    if Trans_flag == 1:

        print("==> Trans-based generation <==")
        generation_type = 'trans'
        transroot = os.path.expanduser(os.path.join(outf) + 'trans/')
        trans_generation(transroot, dataset, batch)

        # # 整体的质量报告
        # print("------------overall quality-----------")
        # trans_png_dataroot = os.path.expanduser(os.path.join(transroot) + 'trans-png/')
        # trans_np_dataroot = os.path.expanduser(os.path.join(transroot) + 'trans-np/')
        # quality_FID(batch, clean_png_root, trans_png_dataroot)
        # quality_IS(dataset, trans_np_dataroot)

        print("==> Method-Guided generation simulation")
        select_num = 500
        data_loader = data_reload(transroot, dataset, generation_type, batch)
        print('Total num:', len(data_loader.dataset), 'Select target num:', 500)
        train_loader, _ = get_dataloader(dataset, data_root, batch)
        print(len(train_loader.dataset))
        # 准备gentle所需信息
        cluster_root = './output/' + dataset + '_' + net_type + '/'
        mean_path = cluster_root + str(cluster_num) + 'classes_mean.npy'
        precision_path = cluster_root + str(cluster_num) + 'classes_precision.npy'
        sample_mean, precision = selection.get_mean_precision(model, net_type, dataset, dataroot, cluster_num,
                                                              batch, train_loader, outf, mean_path, precision_path)
        print("===> Overall-guided")
        str_time = time.time()
        filter_type = 'overall'
        overall_filter_idx = []
        while len(overall_filter_idx) < select_num:
            idx = random.randint(0, len(data_loader.dataset)-1)
            if idx not in overall_filter_idx:
                overall_filter_idx.append(idx)
        print(filter_type, len(overall_filter_idx))
        # 评估loader出数据的质量
        quality_evaluation(transroot, dataset, batch, generation_type, overall_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))

        print("===> Gentle-guided")
        str_time = time.time()
        filter_type = 'gentle'
        gentle_filter_idx = gentle_filter(data_loader, train_loader, model, dataset, cluster_num, sample_mean, precision,
                                          select_num)
        print(filter_type, len(gentle_filter_idx))
        # 评估loader出数据的质量
        quality_evaluation(transroot, dataset, batch, generation_type, gentle_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))

        print("===> Neuron coverage-guided")
        str_time = time.time()
        # 准备 NBC 和 SNAC 的数据
        upper, lower, neuron_num = metrics.get_upper_lower(model, train_loader, dataset)
        nc_k = 0.75
        print('neuron number', neuron_num)
        print('model shape', len(upper))
        for i in upper:
            print('layer:', len(i))

        nac_filter_idx, nbc_filter_idx, snac_filter_idx, tknc_filter_idx = nc_filter(data_loader, batch, model, dataset,
                                                                                     select_num, nc_k, upper, lower)
        filter_type = 'nac'
        print('nac_filter_idx', len(nac_filter_idx))
        quality_evaluation(transroot, dataset, batch, generation_type, nac_filter_idx, filter_type)
        filter_type = 'nbc'
        print('nbc_filter_idx', len(nbc_filter_idx))
        quality_evaluation(transroot, dataset, batch, generation_type, nbc_filter_idx, filter_type)
        filter_type = 'snac'
        print('snac_filter_idx', len(snac_filter_idx))
        quality_evaluation(transroot, dataset, batch, generation_type, snac_filter_idx, filter_type)
        filter_type = 'tknc'
        print('tknc_filter_idx', len(tknc_filter_idx))
        quality_evaluation(transroot, dataset, batch, generation_type, tknc_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))


        print("==> LSC-guided")
        str_time = time.time()
        filter_type = 'lsc'
        lsa_threshold = 0.01
        lsc_train_loader = train_loader
        lsc_data_loader = data_loader
        if dataset == 'mnist' and net_type == 'resnet18':
            lsa_threshold = 0.08
        elif dataset == 'mnist' and net_type == 'vgg11':
            lsa_threshold = 0.008
        elif dataset == 'fashion' and net_type == 'vgg11':
            lsa_threshold = 0.01
        elif dataset == 'fashion' and net_type == 'resnet18':
            lsa_threshold = 0.15
        elif dataset == 'svhn' and net_type == 'vgg16':
            lsa_threshold = 0.003
        elif dataset == 'svhn' and net_type == 'densenet':
            lsa_threshold = 0.03
            batch_lsc = 64
            lsc_train_loader, _ = get_dataloader(dataset, data_root, batch_lsc)
            lsc_data_loader = data_reload(transroot, dataset, generation_type, batch_lsc)
        elif dataset == 'cifar10' and net_type == 'densenet':
            lsa_threshold = 0.01
            batch_lsc = 64
            lsc_train_loader, _ = get_dataloader(dataset, data_root, batch_lsc)
            lsc_data_loader = data_reload(transroot, dataset, generation_type, batch_lsc)
        elif dataset == 'cifar10' and net_type == 'vgg16':
            lsa_threshold = 0.00002
        kdes, removed_cols = get_lsa_kdes(model, lsc_train_loader, lsa_threshold=lsa_threshold)
        lsc_filter_idx = lsc_filter(lsc_data_loader, model, dataset, select_num, kdes, removed_cols)
        print('lsc_filter_idx', len(lsc_filter_idx))
        quality_evaluation(transroot, dataset, batch, generation_type, lsc_filter_idx, filter_type)
        end_time = time.time()
        print("{} Time cost: {:.2f} s.".format(filter_type, end_time - str_time))



        # print("==> NAC-guided")
        # nc_k = 0.75
        # filter_type = 'nac'
        # nac_filter_idx = nac_filter(data_loader, batch, model, dataset, select_num, nc_k)
        # print('nac_filter_idx', len(nac_filter_idx))
        # quality_evaluation(gradroot, dataset, batch, generation_type, nac_filter_idx, filter_type)
        #
        # print("==> NBC-guided")
        # filter_type = 'nbc'
        # nbc_filter_idx = nbc_filter(data_loader, batch, model, dataset, select_num, upper, lower)
        # print('nbc_filter_idx', len(nbc_filter_idx))
        # quality_evaluation(gradroot, dataset, batch, generation_type, nbc_filter_idx, filter_type)
        #
        # print("==> SNAC-guided")
        # filter_type = 'snac'
        # snac_filter_idx = snac_filter(data_loader, batch, model, dataset, select_num, upper)
        # print('snac_filter_idx', len(snac_filter_idx))
        # quality_evaluation(gradroot, dataset, batch, generation_type, snac_filter_idx, filter_type)
        #
        # print("==> TKNC-guided")
        # filter_type = 'tknc'
        # tknc_filter_idx = tknc_filter(data_loader, batch, model, dataset, select_num)
        # print('tknc_filter_idx', len(tknc_filter_idx))
        # quality_evaluation(gradroot, dataset, batch, generation_type, tknc_filter_idx, filter_type)


        # print("==> NAC-guided")
        # nc_k = 0.75
        # filter_type = 'nac'
        # nac_filter_idx = nac_filter(data_loader, batch, model, dataset, select_num, nc_k)
        # print('nac_filter_idx', len(nac_filter_idx))
        # quality_evaluation(ganroot, dataset, batch, generation_type, nac_filter_idx, filter_type)
        #
        # print("==> NBC-guided")
        # filter_type = 'nbc'
        # nbc_filter_idx = nbc_filter(data_loader, batch, model, dataset, select_num, upper, lower)
        # print('nbc_filter_idx', len(nbc_filter_idx))
        # quality_evaluation(ganroot, dataset, batch, generation_type, nbc_filter_idx, filter_type)
        #
        # print("==> SNAC-guided")
        # filter_type = 'snac'
        # snac_filter_idx = snac_filter(data_loader, batch, model, dataset, select_num, upper)
        # print('snac_filter_idx', len(snac_filter_idx))
        # quality_evaluation(ganroot, dataset, batch, generation_type, snac_filter_idx, filter_type)
        #
        # print("==> TKNC-guided")
        # filter_type = 'tknc'
        # tknc_filter_idx = tknc_filter(data_loader, batch, model, dataset, select_num)
        # print('tknc_filter_idx', len(tknc_filter_idx))
        # quality_evaluation(ganroot, dataset, batch, generation_type, tknc_filter_idx, filter_type)

        # print("==> NAC-guided")
        # nc_k = 0.75
        # filter_type = 'nac'
        # nac_filter_idx = nac_filter(data_loader, batch, model, dataset, select_num, nc_k)
        # print('nac_filter_idx', len(nac_filter_idx))
        # quality_evaluation(transroot, dataset, batch, generation_type, nac_filter_idx, filter_type)
        #
        # print("==> NBC-guided")
        # filter_type = 'nbc'
        # nbc_filter_idx = nbc_filter(data_loader, batch, model, dataset, select_num, upper, lower)
        # print('nbc_filter_idx', len(nbc_filter_idx))
        # quality_evaluation(transroot, dataset, batch, generation_type, nbc_filter_idx, filter_type)
        #
        # print("==> SNAC-guided")
        # filter_type = 'snac'
        # snac_filter_idx = snac_filter(data_loader, batch, model, dataset, select_num, upper)
        # print('snac_filter_idx', len(snac_filter_idx))
        # quality_evaluation(transroot, dataset, batch, generation_type, snac_filter_idx, filter_type)
        #
        # print("==> TKNC-guided")
        # filter_type = 'tknc'
        # tknc_filter_idx = tknc_filter(data_loader, batch, model, dataset, select_num)
        # print('tknc_filter_idx', len(tknc_filter_idx))
        # quality_evaluation(transroot, dataset, batch, generation_type, tknc_filter_idx, filter_type)
