from scipy.stats import gaussian_kde
import lib
import numpy as np
import metrics
import My_data_loader
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
from models.densenet import DenseNet3 as cifar_DenseNet3
from models.vgg import VGG as cifar_VGG
from models_mnist.vgg import VGG as mnist_VGG
from models_mnist.resnet import ResNet18 as mnist_ResNet18
from torchvision import datasets
import matplotlib.pyplot as plt
import train_model
import torch.optim as optim
import data_utils
import time
import random
from aug_utils import *
import seaborn as sns
from scipy.stats import wasserstein_distance
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 生成并保存待选集
def target_trans(target):
    return -1


# 待选集：T_train' + OOD
def save_data(dataroot, dataset, save_dir='./data/save'):
    file_name = dataset + '_train_aug_ood'
    img_path = save_dir + '/' + file_name + '_image.npy'
    label_path = save_dir + '/' + file_name + '_label.npy'
    if os.path.exists(img_path) and os.path.exists(label_path):
        return
    data_root = os.path.expanduser(os.path.join(dataroot, str(dataset) + '-data'))
    if dataset == 'cifar10':
        good_aug = datasets.CIFAR10(root=data_root, train=True, download=True, transform=cifar_good_trans)
        bad_aug = datasets.CIFAR10(root=data_root, train=True, download=True, transform=cifar_bad_trans,
                                   target_transform=target_trans)
    elif dataset == 'svhn':
        good_aug = datasets.SVHN(root=data_root, split='train', download=True, transform=svhn_good_trans)
        bad_aug = datasets.SVHN(root=data_root, split='train', download=True, transform=svhn_bad_trans,
                                target_transform=target_trans)
    elif dataset == 'mnist':
        good_aug = datasets.MNIST(root=data_root, train=True, download=True, transform=mnist_good_trans)
        bad_aug = datasets.MNIST(root=data_root, train=True, download=True, transform=mnist_bad_trans,
                                   target_transform=target_trans)
    elif dataset == 'fashion':
        good_aug = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=fashion_good_trans)
        bad_aug = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=fashion_bad_trans,
                                   target_transform=target_trans)
    data = good_aug + bad_aug
    loader = torch.utils.data.DataLoader(data, batch_size=200, shuffle=False, num_workers=1)
    os.makedirs(save_dir, exist_ok=True)
    image, label = [], []
    for data, target in loader:
        image.append(data)
        label.append(target)
    image = np.concatenate(image, axis=0)
    label = np.concatenate(label, axis=0)
    np.save('{}/{}_image'.format(save_dir, file_name), image)
    np.save('{}/{}_label'.format(save_dir, file_name), label)


# 获取待选集的 loader，默认不加原始训练集，重训练时需要加上原始训练集
def get_to_select_loader(image_path, label_path, dataroot, dataset, batch, with_train=False, train_num=None, sampler=None):
    data_root = os.path.expanduser(os.path.join(dataroot, str(dataset) + '-data'))
    tmp_img = np.load(image_path)
    tmp_label = np.load(label_path)
    loader, tmp_data = None, None
    if dataset == 'cifar10':
        tmp_data = data_utils.CIFAR10_local_reload(root=data_root, label=tmp_label, data=tmp_img)
        if with_train:
            ori_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=cifar_train_trans)
            sampler = sampler + list(range(len(tmp_data), len(tmp_data) + train_num))
            loader = torch.utils.data.DataLoader(tmp_data + ori_train, batch_size=batch, shuffle=False, num_workers=2,
                                                 sampler=sampler)
        else:
            loader = torch.utils.data.DataLoader(tmp_data, batch_size=batch, shuffle=False, num_workers=2,
                                                 sampler=sampler)
    elif dataset == 'svhn':
        tmp_data = data_utils.SVHN_local_reload(root=data_root, label=tmp_label, data=tmp_img)
        if with_train:
            ori_train = datasets.SVHN(root=data_root, split='train', download=True, transform=svhn_train_trans)
            sampler = sampler + list(range(len(tmp_data), len(tmp_data) + train_num))
            loader = torch.utils.data.DataLoader(tmp_data + ori_train, batch_size=batch, shuffle=False, num_workers=0,
                                                 sampler=sampler)
        else:
            loader = torch.utils.data.DataLoader(tmp_data, batch_size=batch, shuffle=False, num_workers=0,
                                                 sampler=sampler)
    elif dataset == 'mnist':
        tmp_data = data_utils.MNIST_local_reload(root=data_root, label=tmp_label, data=tmp_img)
        if with_train:
            ori_train = datasets.MNIST(root=data_root, train=True, download=True, transform=mnist_train_trans)
            sampler = sampler + list(range(len(tmp_data), len(tmp_data) + train_num))
            loader = torch.utils.data.DataLoader(tmp_data + ori_train, batch_size=batch, shuffle=False, num_workers=2,
                                                 sampler=sampler)
        else:
            loader = torch.utils.data.DataLoader(tmp_data, batch_size=batch, shuffle=False, num_workers=2,
                                                 sampler=sampler)
    elif dataset == 'fashion':
        tmp_data = data_utils.Fashion_local_reload(root=data_root, label=tmp_label, data=tmp_img)
        if with_train:
            ori_train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=fashion_train_trans)
            sampler = sampler + list(range(len(tmp_data), len(tmp_data) + train_num))
            loader = torch.utils.data.DataLoader(tmp_data + ori_train, batch_size=batch, shuffle=False, num_workers=2,
                                                 sampler=sampler)
        else:
            loader = torch.utils.data.DataLoader(tmp_data, batch_size=batch, shuffle=False, num_workers=2,
                                                 sampler=sampler)
    return loader, len(tmp_data)


# 1. 根据 kernel_in(lrd) 的得分排序，从大到小选择
def get_lrd_kernel(dir_name, in_dataset_name):
    known_val = np.loadtxt('{}/LRD_Val_{}_ID.txt'.format(dir_name, in_dataset_name), delimiter='\n')
    known_test = np.loadtxt('{}/LRD_Test_{}_ID.txt'.format(dir_name, in_dataset_name), delimiter='\n')
    in_score = np.concatenate((known_val, known_test))
    kernel_in = gaussian_kde(in_score)
    return kernel_in


def get_mean_precision(model, net_type, dataset, dataroot, cluster_num, batch, train_loader, outf, mean_path, precision_path):
    if os.path.exists(mean_path) and os.path.exists(precision_path):
        sample_mean = np.load(mean_path, allow_pickle=True)
        precision = np.load(precision_path, allow_pickle=True)
    else:
        if dataset == "cifar10" or dataset == "cifar100":
            trans = cifar_test_trans
        elif dataset == "svhn":
            trans = svhn_test_trans
        elif dataset == 'mnist':
            trans = mnist_test_trans
        elif dataset == 'fashion':
            trans = fashion_test_trans
        feature_list, num_output = metrics.get_information(model, dataset)
        feature_save_path = outf + 'feature_' + net_type + '_' + dataset + '.npy'
        label_save_path = outf + 'label_' + net_type + '_' + dataset + '_' + str(cluster_num) + 'classes.npy'
        feature_size = feature_list[-1]
        metrics.getLastFeatureCluster(model, train_loader, feature_save_path, label_save_path, cluster_num, feature_size)
        c_cluster = np.load(label_save_path)
        # data_root = os.path.expanduser(os.path.join(dataroot, str(dataset) + '-data'))
        train_loader_c, test_loader_c = My_data_loader.getTargetDataSet(dataset, batch, trans, dataroot, c_cluster)

        # prepare for LRD
        sample_mean, precision = lib.sample_estimator(model, cluster_num, feature_list, train_loader_c)
        np.save(mean_path, sample_mean)
        np.save(precision_path, precision)
    return sample_mean, precision


def new_lrd_sampling(loader, train_loader, model, dataset, cluser_num, sample_mean, precision, select_num):
    feature_list, num_output = metrics.get_information(model, dataset)
    lrd_score = lib.get_lrd_score(model, loader, cluser_num, "OOD", sample_mean, precision, num_output - 1)

    M_in_train = lib.get_lrd_score(model, train_loader, cluser_num, True, sample_mean, precision, num_output - 1)
    M_in_train = np.asarray(M_in_train, dtype=np.float32)
    Mahalanobis_in_train = M_in_train.reshape((M_in_train.shape[0], -1))
    Mahalanobis_in_train = np.asarray(Mahalanobis_in_train, dtype=np.float32)
    Mahalanobis_in_train = np.array(Mahalanobis_in_train).flatten()  # score(T_train)

    dis1 = []
    for i in range(len(lrd_score)):
        tmp_score = np.asarray(lrd_score[i])
        tmp_score = np.array(tmp_score).flatten()
        tmp_dis = wasserstein_distance(Mahalanobis_in_train, tmp_score)
        dis1.append(tmp_dis)
    idx_list = np.argsort(dis1)  # dis1 从小到大排序对应的 idx
    select = idx_list[:select_num]
    return select


def get_boundary_lrd(dir_name, in_dataset_name, out_dist_list):
    out_score = np.array([])
    known_val = np.loadtxt('{}/LRD_Val_{}_ID.txt'.format(dir_name, in_dataset_name), delimiter='\n')
    known_test = np.loadtxt('{}/LRD_Test_{}_ID.txt'.format(dir_name, in_dataset_name), delimiter='\n')
    for out_dist in out_dist_list:
        novel_val = np.loadtxt('{}/LRD_Val_{}_OOD.txt'.format(dir_name, out_dist), delimiter='\n')
        novel_test = np.loadtxt('{}/LRD_Test_{}_OOD.txt'.format(dir_name, out_dist), delimiter='\n')
        out_score = np.concatenate((out_score, novel_val))
        out_score = np.concatenate((out_score, novel_test))
    in_score = np.concatenate((known_val, known_test))
    # kernel_in = gaussian_kde(in_score)
    # kernel_out = gaussian_kde(out_score)
    # x = np.linspace(-3000, 1000, 10000)
    # kernel_proportion = kernel_in(x)/kernel_out(x)
    # idx = np.where(kernel_proportion >= 1)[0][0]

    sns.set(color_codes=True)
    sns.kdeplot(in_score, shade=True, label="In")
    sns.kdeplot(out_score, shade=True, label="Out")
    plt.legend()
    plt.show()
    plt.clf()
    # return x[idx]


def lrd_sampling(boundary_score, case_score):
    selected_list = []
    idx_list = np.argsort(case_score)
    for i in idx_list:
        if case_score[i] > boundary_score:
            selected_list.append(i)
    return selected_list


# 2. SE 中的测试用例选择策略
# 2.1 随机采样
def random_sampling(to_select_num, select_num):
    idx_list = np.arange(0, to_select_num)
    np.random.shuffle(idx_list)
    select = idx_list[:select_num]
    return select


# 2.2 按 gini(x) 排序
def gini_sampling(loader, model, net_type, select_num):
    softmax = lib.get_softmax_out(model, loader)
    gini = np.sum(softmax ** 2, axis=1)
    idx_list = np.argsort(gini)  # gini 从小到大排序对应的 idx
    select = idx_list[:select_num]
    return select


# 2.3 MCP 选择方法
def mcp_sampling(loader, model, net_type, select_num):
    dicratio = [[] for i in range(100)]  # max/sec_max 的数值，下标是(10*i+j)，i是 max idx，j是 sec_max idx
    dicindex = [[] for i in range(100)]  # 测试用例对应的 id，下标是(10*i+j)，i是 max idx，j是 sec_max idx
    softmax = lib.get_softmax_out(model, loader)
    tmp_soft = np.sort(softmax)
    tmp_soft_arg = np.argsort(softmax)
    max_index, sec_index = tmp_soft_arg[:, -1], tmp_soft_arg[:, -2]
    ratio = 1.0 * (tmp_soft[:, -1] / tmp_soft[:, -2])  # max/sec_max, 越小越好
    for i in range(len(ratio)):
        dicratio[max_index[i] * 10 + sec_index[i]].append(ratio[i])
        dicindex[max_index[i] * 10 + sec_index[i]].append(i)

    # ----------- selection --------------
    sort_arg, selected_lst = [], []
    for i in range(100):
        tmp_ratio = np.array(dicratio[i])
        tmp_arg = np.argsort(tmp_ratio)  # 每一行按照 ratio 排序后的 idx
        sort_arg.append(tmp_arg)
    selected_num, count = 0, 0
    while selected_num < select_num:
        for i in range(100):
            if len(sort_arg[i]) > count:
                j = sort_arg[i][count]
                selected_lst.append(dicindex[i][j])
                selected_num += 1
                if selected_num == select_num:
                    break
        count += 1  # 每一轮之后 count 增加 1
    return selected_lst


# 2.4 CES 选择方法
def neuron_division(lower, upper, divide):
    divide_list = []
    neurons = len(lower)
    for index in range(neurons):
        interval = np.linspace(lower[index], upper[index], divide)
        divide_list.append(interval)
    return divide_list


def build_neuron_tables(model, loader, divide_list, divide, to_select_num):
    model.eval()
    neurons_num = len(divide_list)
    tab = np.zeros((divide-1, neurons_num))
    for data, target in loader:
        data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
            output, out_features = model.feature_list(data)
        # get hidden features
        # neurons_num = out_features[-1].size(1)  # 最后一层的神经元个数
        for k in range(neurons_num):
            tmp = out_features[-1][:, k]  # 这一批次的数据，对应最后一层第 k 个神经元输出
            tmp = tmp.cpu().numpy()
            for i in range(divide - 1):  # 第 k 个神经元对应的区间划分
                if i == 0:
                    tab[i][k] += np.sum(np.logical_and(tmp >= divide_list[k][i], tmp <= divide_list[k][i + 1]))
                else:
                    tab[i][k] += np.sum(np.logical_and(tmp > divide_list[k][i], tmp <= divide_list[k][i + 1]))
    tab = tab/to_select_num
    return tab


def CES_sampling(model, image_path, label_path, dataroot, dataset, loader, to_select_num, select_num, output_divide=5):
    # 前期工作，准备 to select 待选集合的矩阵
    upper_value_list, lower_value_list, neuron_num = metrics.get_upper_lower(model, loader, dataset)
    last_upper, last_lower = upper_value_list[-1], lower_value_list[-1]
    divide_list = neuron_division(last_lower.numpy(), last_upper.numpy(), output_divide)
    to_select_tab = build_neuron_tables(model, loader, divide_list, output_divide, to_select_num)
    # 准备进入循环
    idx_list = list(range(0, to_select_num))
    random.shuffle(idx_list)
    selected_num, ini, group, max_iter, sel = 30, 30, 30, 30, 330
    selected_id = idx_list[:ini]  # 初始先选30个
    idx_list = list(filter(lambda x: x not in selected_id, idx_list))  # 维护待选集，把已经选了的剔除
    while selected_num < select_num:
        for i in range(max_iter):
            min_ce, min_sel = 10000, None
            for j in range(group):
                random_sample = random.sample(idx_list, sel)  # 从 idx_list 里不放回抽样 sel 个
                tmp_loader, _ = get_to_select_loader(image_path, label_path, dataroot, dataset, sampler=random_sample)
                T_s = build_neuron_tables(model, tmp_loader, divide_list, output_divide, len(random_sample))
                tmp_ce = to_select_tab * np.log(T_s + 1e-5)
                # print("i:", i, " j:", j, " tmp_ce:", tmp_ce, " sum:", np.sum(tmp_ce), " abs sum:", np.abs(np.sum(tmp_ce)))
                if np.abs(np.sum(tmp_ce)) < min_ce:  # 绝对值越小的越好，每次选最小的保留
                    min_ce = np.abs(np.sum(tmp_ce))
                    # min_up, min_low = up_bound, low_bound
                    min_sel = random_sample
            print("min_ce:", min_ce)
            selected_id = selected_id + min_sel
            idx_list = list(filter(lambda x: x not in selected_id, idx_list))  # 维护待选集，把已经选了的剔除
            selected_num = len(selected_id)
            print("len:", len(selected_id), len(set(selected_id)), "idx len:", len(set(idx_list)), "\ntime:",
                  time.asctime(time.localtime(time.time())))
            if selected_num >= select_num:
                break
    return selected_id[:select_num]


# 2.5 LSA 选择方法
def get_ats(model, loader):
    model.eval()
    pred, ats_li = [], []
    for idx, (data, target) in enumerate(loader):
        data = data.cuda()
        outputs = model(data)
        _, predicted = outputs.max(1)
        predicted = predicted.cpu().numpy().tolist()
        pred.extend(predicted)
        with torch.no_grad():
            data = Variable(data)
            output, out_features = model.feature_list(data)

        # 只看最后一层
        for j in range(len(out_features[-1])):  # 第 j 个测试用例
            tmp_out = out_features[-1][j]
            ats_li.append(tmp_out.cpu().numpy())
        # print("idx:", idx)
    ats = np.array([l for l in ats_li if len(l) > 0])
    return ats, pred

def get_ats_without_label(model, loader):
    model.eval()
    pred, ats_li = [], []
    for idx, (data) in enumerate(loader):
        data = data.cuda()
        outputs = model(data)
        _, predicted = outputs.max(1)
        predicted = predicted.cpu().numpy().tolist()
        pred.extend(predicted)
        with torch.no_grad():
            data = Variable(data)
            output, out_features = model.feature_list(data)

        # 只看最后一层
        for j in range(len(out_features[-1])):  # 第 j 个测试用例
            tmp_out = out_features[-1][j]
            ats_li.append(tmp_out.cpu().numpy())
        # print("idx:", idx)
    ats = np.array([l for l in ats_li if len(l) > 0])
    return ats, pred

def get_lsa_kdes(model, train_loader, class_num=10, lsa_threshold=0.001):
    train_ats, train_pred = get_ats(model, train_loader)
    class_matrix, kdes = {}, {}
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
    removed_cols = []
    for label in range(class_num):
        col_vectors = np.transpose(train_ats[class_matrix[label]])
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < lsa_threshold and i not in removed_cols:
                removed_cols.append(i)

    for label in range(class_num):
        refined_ats = np.transpose(train_ats[class_matrix[label]])
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        kdes[label] = (gaussian_kde(refined_ats))
    return kdes, removed_cols


def lsa_sampling(loader, model, select_num, kdes, removed_cols):
    target_ats, target_pred = get_ats(model, loader)
    lsa = []
    for i, at in enumerate(target_ats):
        label = target_pred[i]
        kde = kdes[label]
        refined_at = np.delete(at, removed_cols, axis=0)
        tmp_lsa = np.asscalar(-kde.logpdf(np.transpose(refined_at)))
        lsa.append(tmp_lsa)
    idx_list = np.argsort(lsa)[::-1]  # lsa 从大到小排序对应的 idx
    select = idx_list[:select_num]
    return select


# 3. active learning 相关的选择策略
# 3.1 Least Sampling
def least_sampling(loader, model, select_num):
    softmax = lib.get_softmax_out(model, loader)  # shape: (cases num, class num)
    least = 1 - np.max(softmax, axis=1)  # 每一行（对应每个测试用例）对应的最大值
    idx_list = np.argsort(least)[::-1]  # least maxp 从大到小排序对应的 idx，优先选大的
    select = idx_list[:select_num]
    return select


# 3.2 Margin Sampling
def margin_sampling(loader, model, select_num):
    softmax = lib.get_softmax_out(model, loader)
    tmp_soft = np.sort(softmax)
    margin = tmp_soft[:, -1] - tmp_soft[:, -2]
    idx_list = np.argsort(margin)  # margin 从小到大排序对应的 idx，优先选小的
    select = idx_list[:select_num]
    return select


# 3.3 Entropy Sampling
def entropy_sampling(loader, model, select_num):
    softmax = lib.get_softmax_out(model, loader)
    log_soft = np.log(softmax)
    entropy = (softmax * log_soft).sum(1)
    idx_list = np.argsort(entropy)  # margin 从小到大排序对应的 idx，优先选小的
    select = idx_list[:select_num]
    return select


# 重训练
def retrain_model(net_type, dataset, num_classes, pre_trained_net, save_path, train_loader, test_loader, re_epoch, lr=0.01):
    best_acc = 0


    start_epoch = torch.load(pre_trained_net, map_location="cuda:" + str(0))['epoch']
    print('==> Rebuilding model..')

    if dataset == 'cifar10':
        weight = 5e-4
        # lr = 0.0005
        if net_type == "densenet":
            net = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
        elif net_type == "vgg16":
            net = cifar_VGG('VGG16').to("cuda")
    elif dataset == 'svhn':
        weight = 1e-6
        # lr = 0.01 # 0.005
        if net_type == "densenet":
            net = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
        elif net_type == "vgg16":
            net = cifar_VGG('VGG16').to("cuda")
    elif dataset == 'fashion' or dataset == 'mnist':
        weight = 1e-6
        # lr = 0.01
        if net_type == "resnet18":
            net = mnist_ResNet18(num_c=num_classes).to("cuda")
        elif net_type == "vgg11":
            net = mnist_VGG('VGG11').to("cuda")
    net.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(0))['net'])
    net = net.to(device=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + re_epoch):  # 最多重训练 50 个批次
        train_model.train(epoch, train_loader, net, criterion, optimizer)
        best_acc = train_model.model_test(epoch, save_path, test_loader, net, criterion, best_acc, direct_save=True,
                                          last_epoch=start_epoch + re_epoch - 1)
        # print("best_acc:", best_acc)
        scheduler.step()


def get_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        print("In-task tests:", total, ", Correct tests:", correct, ", Bug tests:", total-correct, ", Acc:", acc)
        # print("acc: ", 100. * correct / total)
    # acc = 100. * correct / total
    return acc


def get_train_and_aug_test(dataroot, dataset, batch, test_num):
    sampler1 = list(range(test_num, test_num * 2))
    # 原始训练集，原始 + 扩增测试集
    aug_trans, test_trans = None, None
    if dataset == 'cifar10':
        test_trans = cifar_test_trans
        aug_trans = cifar_good_trans
    elif dataset == 'svhn':
        test_trans = svhn_test_trans
        aug_trans = svhn_good_trans
    elif dataset == 'mnist':
        test_trans = mnist_test_trans
        aug_trans = mnist_good_trans
    elif dataset == 'fashion':
        test_trans = fashion_test_trans
        aug_trans = fashion_good_trans
    train_loader, test_loader = My_data_loader.getDataSet_with_aug(data_type=dataset, dataroot=dataroot,
                                                                   batch_size=batch, ori_TF=test_trans)
    _, aug_test_loader = My_data_loader.getDataSet_with_aug(data_type=dataset, dataroot=dataroot, batch_size=batch,
                                                            ori_TF=test_trans, test_with_aug=True, aug_TF=aug_trans,
                                                            sampler=sampler1)
    _, test_and_aug_loader = My_data_loader.getDataSet_with_aug(data_type=dataset, dataroot=dataroot, batch_size=batch,
                                                                ori_TF=test_trans, test_with_aug=True, aug_TF=aug_trans)
    return train_loader, test_loader, aug_test_loader, test_and_aug_loader
