import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--dataset", type=str, default='mnist', help="mnist | fashion | svhn")
args = parser.parse_args()
print(args)

os.makedirs("images/" + args.dataset, exist_ok=True)
os.makedirs("images/" + args.dataset + "/clean", exist_ok=True)


def generation_from_generator(gan_save_path, dataset, batch_size, latent_dim):
    gannet = torch.load(gan_save_path, map_location="cuda:0")
    generator_gan = Generator().cuda()
    discriminator_gan = Discriminator().cuda()
    generator_gan.load_state_dict(gannet['generator'])
    discriminator_gan.load_state_dict(gannet['discriminator'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator_gan.cuda()
    discriminator_gan.to(device)
    generator_gan.eval()
    discriminator_gan.eval()
    print(generator_gan)

    print("==>Generating new images...")
    os.makedirs('images/' + dataset + '/test_Gan', exist_ok=True)
    for i in range(100):
        # noises = torch.randn(opt.batch_size, opt.latent_dim, 1, 1)
        noises = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        fake_images = generator_gan(noises)
        save_image(fake_images.detach(), 'images/' + dataset + f'/test_GAN/{i}.png')

print('==> dir is created...')

img_shape = (args.channels, args.img_size, args.img_size)

cuda = True if torch.cuda.is_available() else False

data_root = os.path.expanduser(os.path.join("./data", str(args.dataset) + '-data'))

if args.dataset == 'cifar10' or args.dataset == 'svhn':
    args.latent_dim = 300
elif args.dataset == 'mnist' or args.dataset == 'fashion':
    args.latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

print('==> start to prepare...')
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

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
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
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
print('==> start to train...')

for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

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
        if batches_done % args.sample_interval == 0:
            save_image(real_imgs.data[:25], "images/" + args.dataset + "/clean/%d.png" % batches_done, normalize=True, nrow=5)
            save_image(gen_imgs.data[:25], "images/" + args.dataset + "/%d.png" % batches_done, nrow=5, normalize=True)

    if epoch == 50:
        # os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
        print("==>Saving the bad model...")
        os.makedirs("pre_trained/" + args.dataset, exist_ok=True)
        gan_save_path = "pre_trained/" + args.dataset + "/gan-bad.pth"
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, gan_save_path)

# os.makedirs("images/" + args.dataset + "/model", exist_ok=True)
print("==>Saving the good model...")
os.makedirs("pre_trained/" + args.dataset, exist_ok=True)
gan_save_path = "pre_trained/" + args.dataset + "/gan-good.pth"
torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict()
    }, gan_save_path)
generation_from_generator(gan_save_path, args.dataset, args.batch_size, args.latent_dim)

