# Practical Accuracy Evaluation for Deep Learning Systems via Latent Representation Discrepancy
## Environment

- python=3.6
- Pytorch=1.8.0

- GPU is needed.

You can create the environment through [Conda](https://docs.conda.io/en/latest/):

```shell
conda create -n gentle python=3.6
conda activate gentle
pip install -r requirements.txt
```

## Step 1: Preparing a DNN model as the test object

```shell
python train_model.py --dataset mnist --net_type resnet18 --lr 0.01 --batch_size 64 --gpu 0 --with_aug 0
```

After the training is completed, the output is as follows, and the trained model will be saved in `./pre_trained/mnist_vgg11/mnist_vgg11_aug0.pth`.

For colored dataset (CIFAR-10 and SVHN), we collected additional noise data from other official dataset, and resize them into the same format of our used datsets.
The additional noise datasets could be downloaded from [odin-pytorch](https://github.com/facebookresearch/odin):

- [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
- [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

Please place them to `./data/`.

```shell
python RQ1.py --dataset mnist --net_type resnet18 --gpu 0
```

## Step 2: Conducting Research Questions

### RQ1: Effectiveness.

Firstly, we need to prepare 10 models with different generalizability:

```shell
python train_model.py --dataset mnist --net_type resnet18 --lr 0.01 --batch_size 64 --gpu 0 --with_aug 0.03 0.06 0.09 0.12 0.15 0.18 0.21 0.24 0.27 0.30
```

Then, we can calculate the correlation between Acc/Loss and all baselines:

```shell
python RQ1-effectiveness.py --dataset mnist --net_type resnet18 --gpu 0
```

## RQ2: OOD Detection

We use download links of two out-of-distributin datasets from [odin-pytorch](https://github.com/facebookresearch/odin):

- [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
- [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

Please place them to `./data/`.

```shell
python RQ2-OOD.py --dataset mnist --net_type resnet18 --gpu 0
```

## RQ3: Selection and Retraining

```shell
python RQ3-selection.py --dataset mnist --net_type resnet18 --gpu 0 --re_epoch 5 --batch_size 64
```

