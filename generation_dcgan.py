import argparse
import os
import numpy as np
import math
import sys
from datetime import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--dataset", type=str, default='mnist', help="mnist | fashion | cifar10 | svhn")
args = parser.parse_args()
print(args)

os.makedirs('./print/', exist_ok=True)

now = datetime.now()
strtime = now.strftime('%b%d%H%M')
outfile = './print/' + args.dataset + '_dcgan_'+ strtime +'.txt'
sys.stdout = Logger(outfile)

os.makedirs("images/" + args.dataset, exist_ok=True)
os.makedirs("images/" + args.dataset + "/clean", exist_ok=True)


cuda = True if torch.cuda.is_available() else False
data_root = os.path.expanduser(os.path.join("./data", str(args.dataset) + '-data'))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = args.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))

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
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
# Configure data loader
if args.dataset == 'mnist':
    # os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
elif args.dataset == 'fashion':
    # os.makedirs("../../data/fashion", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
elif args.dataset == 'cifar10':
    # os.makedirs("../../data/cifar10", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(args.img_size),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
elif args.dataset == 'svhn':
    # os.makedirs("../../data/svhn", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root=data_root,
            split='train',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                # transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
            ]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
print('==> data is loaded...')

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        if i % 100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        batches_done = epoch * len(dataloader) + i
        # if epoch % 10 == 0:
        #     save_image(gen_imgs.data[:25], "images/" + args.dataset + "/epoch_%d.png" % epoch, nrow=5, normalize=True)
        if batches_done % args.sample_interval == 0:
            save_image(real_imgs.data[:25], "images/" + args.dataset + "/clean/%d.png" % batches_done, normalize=True,
                       nrow=5)
            save_image(gen_imgs.data[:25], "images/" + args.dataset + "/%d.png" % batches_done, nrow=5, normalize=True)


    if epoch == 50:
        # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
        print("==>Saving the bad model...")
        os.makedirs("pre_trained/" + args.dataset, exist_ok=True)
        gan_save_path = "pre_trained/" + args.dataset + "/dcgan-50epoch.pth"
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, gan_save_path)
    elif epoch == 100:
        # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
        print("==>Saving the bad model...")
        os.makedirs("pre_trained/" + args.dataset, exist_ok=True)
        gan_save_path = "pre_trained/" + args.dataset + "/dcgan-100epoch.pth"
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, gan_save_path)
    elif epoch == 200:
        # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
        print("==>Saving the bad model...")
        os.makedirs("pre_trained/" + args.dataset, exist_ok=True)
        gan_save_path = "pre_trained/" + args.dataset + "/dcgan-200epoch.pth"
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, gan_save_path)
    elif epoch == 300:
        # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
        print("==>Saving the bad model...")
        os.makedirs("pre_trained/" + args.dataset, exist_ok=True)
        gan_save_path = "pre_trained/" + args.dataset + "/dcgan-300epoch.pth"
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, gan_save_path)

# os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
print("==>Saving the good model...")
os.makedirs("pre_trained/" + args.dataset, exist_ok=True)
gan_save_path = "pre_trained/" + args.dataset + "/dcgan-400epoch.pth"
torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict()
}, gan_save_path)
# generation_from_generator(gan_save_path, args.dataset, args.batch_size, args.latent_dim)

#
# print("================================")
# print("START TO TRAIN THE CIFAR10 MODEL")
# print("START TO TRAIN THE CIFAR10 MODEL")
# print("START TO TRAIN THE CIFAR10 MODEL")
# print("================================")
#
#
# ## cifar10
# dataset = 'cifar10'
#
# os.makedirs("images/" + args.dataset, exist_ok=True)
# os.makedirs("images/" + args.dataset + "/clean", exist_ok=True)
#
# cuda = True if torch.cuda.is_available() else False
# data_root = os.path.expanduser(os.path.join("./data", str(dataset) + '-data'))
#
# # Loss function
# adversarial_loss = torch.nn.BCELoss()
#
# # Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
#
# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
#
# # Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)
#
# dataloader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(
#         root=data_root,
#         train=True,
#         download=True,
#         transform=transforms.Compose([
#             transforms.Resize(args.img_size),
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#             # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ]),
#     ),
#     batch_size=args.batch_size,
#     shuffle=True,
# )
#
# print('==> data is loaded...')
#
# # Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
#
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
# # ----------
# #  Training
# # ----------
#
# for epoch in range(args.n_epochs):
#     for i, (imgs, _) in enumerate(dataloader):
#
#         # Adversarial ground truths
#         valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
#         fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
#
#         # Configure input
#         real_imgs = Variable(imgs.type(Tensor))
#
#         # -----------------
#         #  Train Generator
#         # -----------------
#
#         optimizer_G.zero_grad()
#
#         # Sample noise as generator input
#         z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
#
#         # Generate a batch of images
#         gen_imgs = generator(z)
#
#         # Loss measures generator's ability to fool the discriminator
#         g_loss = adversarial_loss(discriminator(gen_imgs), valid)
#
#         g_loss.backward()
#         optimizer_G.step()
#
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#
#         optimizer_D.zero_grad()
#
#         # Measure discriminator's ability to classify real from generated samples
#         real_loss = adversarial_loss(discriminator(real_imgs), valid)
#         fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
#         d_loss = (real_loss + fake_loss) / 2
#
#         d_loss.backward()
#         optimizer_D.step()
#
#         if i % 100 == 0:
#             print(
#                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                 % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#             )
#
#         if epoch % 10 == 0:
#             save_image(gen_imgs.data[:25], "images/" + dataset + "/epoch_%d.png" % epoch, nrow=5, normalize=True)
#         batches_done = epoch * len(dataloader) + i
#         if batches_done % args.sample_interval == 0:
#             save_image(real_imgs.data[:25], "images/" + dataset + "/clean/%d.png" % batches_done, normalize=True,
#                        nrow=5)
#             save_image(gen_imgs.data[:25], "images/" + dataset + "/%d.png" % batches_done, nrow=5, normalize=True)
#
#     if epoch == 50:
#         # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
#         print("==>Saving the bad model...")
#         os.makedirs("pre_trained/" + dataset, exist_ok=True)
#         gan_save_path = "pre_trained/" + dataset + "/dcgan-50epoch.pth"
#         torch.save({
#             'generator': generator.state_dict(),
#             'discriminator': discriminator.state_dict()
#         }, gan_save_path)
#     if epoch == 100:
#         # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
#         print("==>Saving the bad model...")
#         os.makedirs("pre_trained/" + dataset, exist_ok=True)
#         gan_save_path = "pre_trained/" + dataset + "/dcgan-100epoch.pth"
#         torch.save({
#             'generator': generator.state_dict(),
#             'discriminator': discriminator.state_dict()
#         }, gan_save_path)
#     if epoch == 200:
#         # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
#         print("==>Saving the bad model...")
#         os.makedirs("pre_trained/" + dataset, exist_ok=True)
#         gan_save_path = "pre_trained/" + dataset + "/dcgan-200epoch.pth"
#         torch.save({
#             'generator': generator.state_dict(),
#             'discriminator': discriminator.state_dict()
#         }, gan_save_path)
#     if epoch == 300:
#         # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
#         print("==>Saving the bad model...")
#         os.makedirs("pre_trained/" + dataset, exist_ok=True)
#         gan_save_path = "pre_trained/" + dataset + "/dcgan-300epoch.pth"
#         torch.save({
#             'generator': generator.state_dict(),
#             'discriminator': discriminator.state_dict()
#         }, gan_save_path)
#
# # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
# print("==>Saving the good model...")
# os.makedirs("pre_trained/" + dataset, exist_ok=True)
# gan_save_path = "pre_trained/" + dataset + "/dcgan-400epoch.pth"
# torch.save({
#     'generator': generator.state_dict(),
#     'discriminator': discriminator.state_dict()
# }, gan_save_path)
# # generation_from_generator(gan_save_path, args.dataset, args.batch_size, args.latent_dim)
