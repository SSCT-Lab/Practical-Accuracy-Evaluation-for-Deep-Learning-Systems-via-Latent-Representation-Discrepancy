import metrics
import argparse
import os
import torch
import My_data_loader
from models.densenet import DenseNet3 as cifar_DenseNet3
from models.vgg import VGG as cifar_VGG
from models_mnist.vgg import VGG as mnist_VGG
from models_mnist.resnet import ResNet18 as mnist_ResNet18
from aug_utils import *
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr
import numpy as np
from selection import *
import sys
from datetime import datetime
import metrics_detection
import lib

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_nc(model, loader, upper_value_list, lower_value_list, neuron_num, nc_k, dataset):
    upper, lower, act = metrics.calc_nc(model, loader, upper_value_list, lower_value_list, nc_k, dataset)
    nbc = (upper + lower) / (2 * neuron_num)
    snac = upper / neuron_num
    nac = act / neuron_num
    tknc = metrics.calc_tknc(model, loader, dataset)
    print("nbc:", nbc, "snac:", snac, "nac:", nac, "tknc:", tknc)
    return nbc, snac, nac, tknc


def print_correlation(eg, nbc, snac, nac, tknc, lrd, acc, lsc):
    print("-------------------------")
    print("EG:", eg)
    print("NBC_gen:", nbc)
    print("SNAC_gen:", snac)
    print("NAC_gen:", nac)
    print("TKNC_gen:", tknc)
    print("LSC_gen:", lsc)
    print("LRD_gen:", lrd)
    print("ACC_gen:", acc)
    print("-------------------------")
    print("Spearman Relation between EG and LRD:", spearmanr(eg, lrd))
    print("Spearman Relation between EG and NBC:", spearmanr(eg, nbc))
    print("Spearman Relation between EG and SNAC:", spearmanr(eg, snac))
    print("Spearman Relation between EG and NAC:", spearmanr(eg, nac))
    print("Spearman Relation between EG and TKNC:", spearmanr(eg, tknc))
    print("Spearman Relation between EG and LSC:", spearmanr(eg, lsc))
    print("-------------------------")
    print("Spearman Relation between ACC and LRD:", spearmanr(acc, lrd))
    print("Spearman Relation between ACC and NBC:", spearmanr(acc, nbc))
    print("Spearman Relation between ACC and SNAC:", spearmanr(acc, snac))
    print("Spearman Relation between ACC and NAC:", spearmanr(acc, nac))
    print("Spearman Relation between ACC and TKNC:", spearmanr(acc, tknc))
    print("Spearman Relation between ACC and LSC:", spearmanr(acc, lsc))
    print("-------------------------")

def save_data(loader, file_name):
    os.makedirs('data/save', exist_ok=True)
    image, label = [], []
    for data, target in loader:
        image.append(data)
        label.append(target)
    image = np.concatenate(image, axis=0)
    label = np.concatenate(label, axis=0)
    np.save('./data/save/{}_image'.format(file_name), image)
    np.save('./data/save/{}_label'.format(file_name), label)


def correlation(batch_size, dataset, dataroot, pre_trained_model, net_type, outf, gpu, trans, aug_transform):
    train_num, test_num, num_classes, cluster_num, nc_k = 50000, 10000, 10, 10, 0.75
    if dataset == "svhn":
        train_num, test_num = 73257, 26032
    if dataset == "cifar100":
        num_classes = 100
    if dataset == 'mnist' or dataset == 'fashion':
        train_num = 60000
    EG_gen, NBC_gen, SNAC_gen, NAC_gen, TKNC_gen, LRD_gen, ACC_gen, LSC_gen = [], [], [], [], [], [], [], []

    sampler1 = list(range(test_num, test_num * 2))  # test set: aug dataset
    sampler2 = list(range(train_num, train_num * 2))  # train set: aug dataset
    train_loader, aug_test_loader = My_data_loader.getDataSet_with_aug(data_type=dataset, dataroot=dataroot,
                                                                       batch_size=batch_size, ori_TF=trans,
                                                                       test_with_aug=True, aug_TF=aug_transform, sampler=sampler1)
    aug_train_loader, test_loader = My_data_loader.getDataSet_with_aug(data_type=dataset, dataroot=dataroot,
                                                                       batch_size=batch_size, ori_TF=trans,
                                                                       train_with_aug=True, aug_TF=aug_transform, sampler=sampler2)
    train_and_aug_loader, _ = My_data_loader.getDataSet_with_aug(data_type=dataset, dataroot=dataroot,
                                                                 batch_size=batch_size, ori_TF=trans,
                                                                 train_with_aug=True, aug_TF=aug_transform)
    train_and_ood_loader = My_data_loader.get_OOD_Dataset(dataset, batch_size, trans, dataroot, True)

    outf = outf + net_type + '_' + dataset + '/'
    os.makedirs(outf, exist_ok=True)

    aug_range = list(range(3, 31, 3))
    model_files = [str(net_type) + "_" + str(dataset) + "_aug" + str(i) for i in aug_range]

    for model_name in model_files:
        pre_trained_net = pre_trained_model + '/' + model_name + '.pth'
        if dataset == 'cifar10' or dataset == 'svhn':
            if net_type == "densenet":
                model = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
            elif net_type == "vgg16":
                model = cifar_VGG('VGG16').to("cuda")
        if dataset == 'fashion' or dataset == 'mnist':
            if net_type == "resnet18":
                model = mnist_ResNet18(num_c=num_classes).to("cuda")
            elif net_type == "vgg11":
                model = mnist_VGG('VGG11').to("cuda")

        cudnn.benchmark = True
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(gpu))['net'])
        model.cuda()
        EG = metrics.cacl_EG(model, net_type, aug_test_loader)

        ACC = get_acc(model, aug_test_loader)
        print("ACC", ACC)
        # LSC prepare
        if dataset == 'mnist' and net_type == 'resnet18':
            lsa_threshold = 0.08
        elif dataset == 'mnist' and net_type == 'vgg11':
            lsa_threshold = 0.006
        elif dataset == 'fashion' and net_type == 'vgg11':
            lsa_threshold = 0.01
        elif dataset == 'fashion' and net_type == 'resnet18':
            lsa_threshold = 0.15
        elif dataset == 'svhn' and net_type == 'vgg16':
            lsa_threshold = 0.003
        elif dataset == 'svhn' and net_type == 'densenet':
            lsa_threshold = 0.03
            batch_lsc = 64
        elif dataset == 'cifar10' and net_type == 'densenet':
            lsa_threshold = 0.01
            batch_lsc = 64
        elif dataset == 'cifar10' and net_type == 'vgg16':
            lsa_threshold = 0.00002
        kdes, removed_cols = get_lsa_kdes(model, train_loader, lsa_threshold=lsa_threshold)

        upper_value_list, lower_value_list, neuron_num = metrics.get_upper_lower(model, train_loader, dataset)
        print("train_and_aug:")
        nbc_train_and_aug, snac_train_and_aug, nac_train_and_aug, tknc_train_and_aug = \
            get_nc(model, train_and_aug_loader, upper_value_list, lower_value_list, neuron_num, nc_k, dataset)
        lsc_train_and_aug = get_lsc(model, train_and_aug_loader, dataset, kdes, removed_cols)

        print("train_and_ood:")
        nbc_train_and_ood, snac_train_and_ood, nac_train_and_ood, tknc_train_and_ood = \
            get_nc(model, train_and_ood_loader, upper_value_list, lower_value_list, neuron_num, nc_k, dataset)
        lsc_train_and_ood = get_lsc(model, train_and_ood_loader, dataset, kdes, removed_cols)

        print("train:")
        nbc_train, snac_train, nac_train, tknc_train = get_nc(model, train_loader, upper_value_list, lower_value_list,
                                                              neuron_num, nc_k, dataset)
        lsc_train = get_lsc(model, train_loader, dataset, kdes, removed_cols)

        print("------ ", pre_trained_net, " ------")
        print("Empirical Generalization (EG) = ", EG)
        NBC = 1-((nbc_train_and_aug - nbc_train) / (nbc_train_and_ood - nbc_train))
        SNAC = 1-((snac_train_and_aug - snac_train) / (snac_train_and_ood - snac_train))
        if nac_train_and_ood - nac_train == 0:
            NAC = "NAN"
        else:
            NAC = 1-((nac_train_and_aug - nac_train) / (nac_train_and_ood - nac_train))
        if tknc_train_and_ood - tknc_train == 0:
            TKNC = "NAN"
        else:
            TKNC = 1-((tknc_train_and_aug - tknc_train) / (tknc_train_and_ood - tknc_train))
        if lsc_train_and_ood - lsc_train == 0:
            LSC = "NAN"
        else:
            LSC = 1-((lsc_train_and_aug - lsc_train) / (lsc_train_and_ood - lsc_train))

        info = model_name.split(".")[0]
        feature_save_path = './data/' + net_type + '_' + dataset + '_feature_' + info + '.npy'
        label_save_path = './data/' + net_type + '_' + dataset + '_label_' + info + '.npy'
        lrd = metrics.calc_lrd(model, train_loader, dataset, batch_size, trans, dataroot, aug_train_loader,
                                     feature_save_path, label_save_path, cluster_num)
        print("LRD: ", lrd)
        print("NBC: ", NBC, "\nSNAC: ", SNAC, " \nNAC: ", NAC, "\nTKNC: ", TKNC, "\nLSC: ", LSC)

        EG_gen.append(EG)
        NBC_gen.append(NBC)
        SNAC_gen.append(SNAC)
        NAC_gen.append(NAC)
        TKNC_gen.append(TKNC)
        LRD_gen.append(lrd)
        LSC_gen.append(LSC)
        ACC_gen.append(ACC)

    return EG_gen, NBC_gen, SNAC_gen, NAC_gen, TKNC_gen, LRD_gen, ACC_gen, LSC_gen


def get_lsc(model, data_loader, dataset, kdes, removed_cols):
    lsc_data_loader = data_loader
    target_ats, target_pred = get_ats(model, lsc_data_loader)
    lsa = []
    for i, at in enumerate(target_ats):
        label = target_pred[i]
        kde = kdes[label]
        refined_at = np.delete(at, removed_cols, axis=0)
        tmp_lsa = np.asscalar(-kde.logpdf(np.transpose(refined_at)))
        lsa.append(tmp_lsa)
    if dataset == 'mnist' or dataset == 'fashion':
        ub, n = 2000, 1000
    elif dataset == 'cifar10' or dataset == 'svhn':
        ub, n = 100, 1000
    bucket_l = ub / n
    covered_lsc = [0] * n
    for i in range(n):
        lower = bucket_l * i
        upper = bucket_l * (i+1)
        for j in range(len(lsa)):
            if lsa[j] > lower and lsa[j] <= upper:
                covered_lsc[i] = 1
    lsc = sum(covered_lsc) / n
    print('lsc: ', lsc)
    return lsc

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

    os.makedirs('./print/', exist_ok=True)

    now = datetime.now()
    strtime = now.strftime('%b%d%H%M')
    outfile = './print/' + args.dataset + '_' + args.net_type + '_rq1_' + strtime + '.txt'
    sys.stdout = Logger(outfile)

    if args.dataset == 'cifar10':
        trans, aug_transform = cifar_test_trans, cifar_good_trans
    elif args.dataset == 'svhn':
        trans, aug_transform = svhn_test_trans, svhn_good_trans
    elif args.dataset == 'mnist':
        trans, aug_transform = mnist_test_trans, mnist_good_trans
    elif args.dataset == 'fashion':
        trans, aug_transform = fashion_test_trans, fashion_good_trans

    EG_tot, NBC_tot, SNAC_tot, NAC_tot, TKNC_tot, LRD_tot, ACC_tot, LSC_tot = [], [], [], [], [], [], [], []
    maxp_tot, oe_tot, ODIN_tot, energy_tot = [], [], [], []
    pre_trained_path = args.pre_trained_model + '/' + args.net_type + '_' + args.dataset + '/'
    if args.dataset == 'cifar10' and args.net_type == 'densenet':
        pre_trained_path = args.pre_trained_model + '/' + args.net_type + '_' + args.dataset + '/New Folder/'
    for i in range(3):
        EG, NBC_gen, SNAC_gen, NAC_gen, TKNC_gen, LRD_gen, ACC_gen, LSC_gen = correlation(args.batch_size, args.dataset,
                                                                           args.dataroot, pre_trained_path,
                                                                           args.net_type, args.outf, args.gpu, trans,
                                                                           aug_transform)
        print("------------ times: ", i, " ------------")
        print_correlation(EG, NBC_gen, SNAC_gen, NAC_gen, TKNC_gen, LRD_gen, ACC_gen, LSC_gen)
        EG_tot.extend(EG)
        NBC_tot.extend(NBC_gen)
        SNAC_tot.extend(SNAC_gen)
        NAC_tot.extend(NAC_gen)
        TKNC_tot.extend(TKNC_gen)
        LRD_tot.extend(LRD_gen)
        ACC_tot.extend(ACC_gen)
        LSC_tot.extend(LSC_gen)
    print("----------- final results --------------")
    print("Total number:", len(EG_tot))
    print_correlation(EG_tot, NBC_tot, SNAC_tot, NAC_tot, TKNC_tot, LRD_tot, ACC_tot, LSC_tot)


