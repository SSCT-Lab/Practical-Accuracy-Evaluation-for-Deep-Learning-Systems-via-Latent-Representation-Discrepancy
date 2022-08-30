import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import lib
import My_data_loader
import metrics
import os
from selection import *


def get_lrd_score(dataset, net_type, score_path, num_classes=10):
    loader_batch, test_num = 200, 10000
    if dataset == 'svhn':
        test_num = 26032
        trans = svhn_test_trans
    if dataset == 'cifar10':
        trans = cifar_test_trans
    if dataset == 'mnist':
        trans = mnist_test_trans
    if dataset == 'fashion':
        trans = fashion_test_trans
    if dataset == 'cifar10' or dataset == 'svhn':
        if net_type == "densenet":
            model = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
            loader_batch = 100
        elif net_type == "vgg16":
            model = cifar_VGG('VGG16').to("cuda")
    if dataset == 'fashion' or dataset == 'mnist':
        if net_type == "resnet18":
            model = mnist_ResNet18(num_c=num_classes).to("cuda")
        elif net_type == "vgg11":
            model = mnist_VGG('VGG11').to("cuda")

    pre_trained_dir = "./pre_trained/" + net_type + "_" + dataset + "/"
    pre_trained_net = pre_trained_dir + "/" + net_type + "_" + dataset + "_aug0.pth"
    model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(0))['net'])
    model.cuda()
    train_loader, test_loader, aug_test_loader, test_and_aug_loader = get_train_and_aug_test(dataroot, dataset,
                                                                                             loader_batch, test_num)
    mean_path = outf + str(cluster_num) + 'classes_mean.npy'
    precision_path = outf + str(cluster_num) + 'classes_precision.npy'
    sample_mean, precision = get_mean_precision(model, net_type, in_dataset_name, dataroot, cluster_num, 200,
                                                train_loader, outf, mean_path, precision_path)
    feature_list, num_output = metrics.get_information(model, in_dataset_name)
    lib.get_lrd_score(model, test_loader, cluster_num, "ID", sample_mean, precision, num_output - 1,
                         write_file=True, dataset_name=in_dataset_name, outf=score_path)
    for out_dist in out_dist_list:
        out_test_loader = My_data_loader.getNonTargetDataSet(out_dist, 200, trans, dataroot)
        print('Out-distribution: ' + out_dist)
        lib.get_lrd_score(model, out_test_loader, cluster_num, "OOD", sample_mean, precision,
                             num_output - 1, write_file=True, dataset_name=out_dist, outf=score_path)


def dke_plot(dir_name, in_dataset_name, out_dist_list):
    if not os.path.exists('{}/LRD_Val_{}_OOD.txt'.format(dir_name, out_dist_list[0])):  # OOD的 LRD得分没有保存，需要先计算一次
        get_lrd_score(in_dataset_name, net_type, score_path, num_classes=10)

    out_score = np.array([])
    known_val = np.loadtxt('{}/LRD_Val_{}_ID.txt'.format(dir_name, in_dataset_name), delimiter='\n')
    known_test = np.loadtxt('{}/LRD_Test_{}_ID.txt'.format(dir_name, in_dataset_name), delimiter='\n')
    for out_dist in out_dist_list:
        novel_val = np.loadtxt('{}/LRD_Val_{}_OOD.txt'.format(dir_name, out_dist), delimiter='\n')
        novel_test = np.loadtxt('{}/LRD_Test_{}_OOD.txt'.format(dir_name, out_dist), delimiter='\n')
        out_score = np.concatenate((out_score, novel_val))
        out_score = np.concatenate((out_score, novel_test))
    in_score = np.concatenate((known_val, known_test))

    sns.set(color_codes=True)
    sns.kdeplot(in_score, shade=True, label="In")
    sns.kdeplot(out_score, shade=True, label="Out")
    plt.legend()
    plt.show()
    plt.clf()


if __name__ == '__main__':
    model_li = ['mnist_resnet18', 'mnist_vgg11', 'fashion_resnet18', 'fashion_vgg11',
                'cifar10_densenet', 'cifar10_vgg16', 'svhn_vgg16']  # 'svhn_densenet'
    for item in model_li:
        score_path = './rq1_results/' + item + '/'
        os.makedirs(score_path, exist_ok=True)
        in_dataset_name = item.split('_')[0]
        net_type = item.split('_')[1]
        outf = './output/' + in_dataset_name + '_' + net_type + '/'
        out_dist_list = None
        cluster_num = 10
        dataroot = './data'
        if in_dataset_name == 'svhn':
            out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        elif in_dataset_name == "cifar10":
            out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        elif in_dataset_name == "mnist":
            out_dist_list = ['emnist_letters', 'kmnist', 'fashion']
        elif in_dataset_name == "fashion":
            out_dist_list = ['mnist', 'kmnist', 'emnist_letters']

        dke_plot(score_path, in_dataset_name, out_dist_list)