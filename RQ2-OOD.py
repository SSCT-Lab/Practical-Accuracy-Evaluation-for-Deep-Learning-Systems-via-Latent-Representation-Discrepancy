import argparse
import os
from models.densenet import DenseNet3 as cifar_DenseNet3
from models.vgg import VGG as cifar_VGG
from models_mnist.vgg import VGG as mnist_VGG
from models_mnist.resnet import ResNet18 as mnist_ResNet18
import My_data_loader
import torch
import metrics_detection
from aug_utils import *
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lib


def detection_result_tocsv(csv_path, out_dist_list, maxp, oe, ODIN, energy, lrd):
    f = open(csv_path, 'w', newline='')
    fieldnames = ['Method', 'Out', 'TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT', 'FPR']
    methods = ['maxp', 'oe', 'ODIN', 'energy', 'lrd']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for method in methods:
        tmp = eval(method)
        for i in range(len(tmp)):
            tmp[i]['Test']['Method'] = method
            tmp[i]['Test']['Out'] = out_dist_list[i]
            print(tmp[i])
            writer.writerow(tmp[i]['Test'])
    f.close()


def dke_plot(dir_name, in_dataset_name, net_type, out_dist_list, energy_temperature, ODIN_temperature, lrd_cluster):
    lib.get_score(model, test_loader, energy_temperature[0], ODIN_temperature[0], dir_name, "ID", dataset)
    out_dist = out_dist_list[0]
    out_test_loader = My_data_loader.getNonTargetDataSet(out_dist, 200, trans, dataroot)
    print('Out-distribution: ' + out_dist)
    print("Len:", len(out_test_loader))
    lib.get_score(model, out_test_loader, energy_temperature[0], ODIN_temperature[0], dir_name, "OOD", dataset)
    metrics_detection.get_lrd_performance(model, net_type, dataset, dir_name, batch_size, dataroot, train_loader,
                                             test_loader, out_dist_list, trans, cluster_num_list=[lrd_cluster[0]])

    methods = ['maxp', 'oe', 'ODIN', 'energy', 'lrd']
    for method in methods:
        if method == 'lrd':
            known_val = np.loadtxt('{}/LRD_Val_{}_ID.txt'.format(dir_name, in_dataset_name), delimiter='\n')
            known_test = np.loadtxt('{}/LRD_Test_{}_ID.txt'.format(dir_name, in_dataset_name), delimiter='\n')
            out_dist = out_dist_list[0]
            novel_val = np.loadtxt('{}/LRD_Val_{}_OOD.txt'.format(dir_name, out_dist), delimiter='\n')
            novel_test = np.loadtxt('{}/LRD_Test_{}_OOD.txt'.format(dir_name, out_dist), delimiter='\n')
            out_score = np.concatenate((novel_val, novel_test))
            in_score = np.concatenate((known_val, known_test))
        else:
            in_score = np.loadtxt('{}/{}_{}_ID.txt'.format(dir_name, method, in_dataset_name), delimiter='\n')
            out_score = np.loadtxt('{}/{}_{}_OOD.txt'.format(dir_name, method, in_dataset_name), delimiter='\n')

        sns.set(color_codes=True)
        sns.kdeplot(in_score, shade=True, label="In-task")
        sns.kdeplot(out_score, shade=True, label="Out-of-task")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./rq1_results/fig/rq1_{in_dataset_name}_{net_type}_{method}.pdf', dpi=200)
        plt.clf()


# LRD 和一些 OOD检测方法（maxp, ODIN, energy, OE, Mahala, NBC, SNAC）的 OOD检测效果（ACC, AUROC, TNR95）
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch code: Train the models.')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataset', required=True, help='cifar10 | svhn | mnist | fashion')
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--pre_trained_model', default='./pre_trained', help='path to pre_trained_models')
    parser.add_argument('--net_type', required=True, help='resnet18 | vgg16 | densenet | vgg16')
    parser.add_argument('--outf', default='./output/', help='folder to output results')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    args = parser.parse_args()
    print(args)

    model, out_dist_list, trans = None, None, None
    batch_size, dataset, net_type, dataroot = args.batch_size, args.dataset, args.net_type, args.dataroot
    train_num, test_num, num_classes = 50000, 10000, 10
    if dataset == 'svhn':
        if net_type == "densenet":
            model = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
        elif net_type == "vgg16":
            model = cifar_VGG('VGG16').to("cuda")
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        train_num, test_num = 73257, 26032
        trans = svhn_test_trans
    elif dataset == "cifar10":
        if net_type == "densenet":
            model = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
        elif net_type == "vgg16":
            model = cifar_VGG('VGG16').to("cuda")
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        trans = cifar_test_trans
    elif dataset == "mnist":
        if net_type == "vgg11":
            model = mnist_VGG('VGG11').to("cuda")
        elif net_type == "resnet18":
            model = mnist_ResNet18(num_c=num_classes).to("cuda")
        train_num = 60000
        out_dist_list = ['kmnist', 'fashion', 'emnist_letters']
        trans = mnist_test_trans
    elif dataset == "fashion":
        if net_type == "vgg11":
            model = mnist_VGG('VGG11').to("cuda")
        elif net_type == "resnet18":
            model = mnist_ResNet18(num_c=num_classes).to("cuda")
        train_num = 60000
        out_dist_list = ['kmnist', 'mnist', 'emnist_letters']
        trans = fashion_test_trans

    outf = args.outf + dataset + '_' + net_type + '/'
    os.makedirs(outf, exist_ok=True)

    pre_trained_path = args.pre_trained_model + '/' + args.net_type + '_' + args.dataset + '/'
    pre_trained_net = pre_trained_path + '/' + args.net_type + "_" + args.dataset + "_aug0.pth"
    model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu))['net'])
    model.cuda()
    train_loader, test_loader = My_data_loader.getDataSet_with_aug(data_type=dataset, dataroot=dataroot,
                                                                   batch_size=batch_size, ori_TF=trans)

    # get maxp, ODIN, OE. Energy results
    maxp_result, oe_result, ODIN_result, ODIN_temperature, energy_result, energy_temperature = \
        metrics_detection.get_performance(model, net_type, dataset, outf, batch_size, dataroot,
                                          test_loader, out_dist_list, trans)
    print("------- Dataset:", dataset, "Model:", net_type, "-------")
    print("------- maxp results -------")
    metrics_detection.show_detection_score(maxp_result, out_dist_list)
    print("------- OE results -------")
    metrics_detection.show_detection_score(oe_result, out_dist_list)
    print("------- energy results -------")
    metrics_detection.show_detection_score(energy_result, out_dist_list, best_temperature=energy_temperature)
    print("------- ODIN results -------")
    metrics_detection.show_detection_score(ODIN_result, out_dist_list, best_temperature=ODIN_temperature)

    # get LRD score results
    cluster_num_list = [8, 9, 10, 11, 12, 13, 14, 15]
    LRD_result, LRD_cluster = metrics_detection.get_lrd_performance(model, net_type, dataset, outf, batch_size,
                                                                             dataroot, train_loader, test_loader,
                                                                             out_dist_list, trans, cluster_num_list)
    print("------- LRD results -------")
    metrics_detection.show_detection_score(LRD_result, out_dist_list, best_cluster=LRD_cluster)

    # save results to CSV file
    save_path = 'rq1_{}_{}.csv'.format(args.dataset, args.net_type)
    detection_result_tocsv(save_path, out_dist_list, maxp_result, oe_result, ODIN_result, energy_result, LRD_result)

    # save the KDE distribution as PDF file
    plot_dir = "./rq1_results/" + dataset + '_' + net_type + '/'
    os.makedirs(plot_dir, exist_ok=True)
    dke_plot(plot_dir, dataset, net_type, out_dist_list, energy_temperature=energy_temperature,
             ODIN_temperature=ODIN_temperature, LRD_cluster=LRD_cluster)