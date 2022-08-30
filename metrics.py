'''
calculate the Empirical Generalization (EG), NBC, SNAC, LRD
EG: input a pre-trained model, train dataset, and test dataset, output the EG for this model;
Others: input a pre-trained model, and train dataset, output the generalization score for this model.
'''
import torch
import My_data_loader
import models
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import lib
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import os


def cacl_EG(model, net_type, aug_test_loader):
    model.eval()
    EG_results = []
    for data, targets in aug_test_loader:
        data = data.cuda()
        data = Variable(data, requires_grad=True)
        batch_output = model(data)
        soft_out = F.softmax(batch_output, dim=1)  # sum=1, softmax output vector
        for i in range(len(soft_out)):
            EG_results.append(1 - soft_out[i][targets[i]].data.cpu().numpy())  # 1-y(t)

    EG = np.mean(EG_results)
    return EG


def cacl_avg_MSP(model, net_type, test_loader):
    magnitude = 0
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    MSP_results = []
    for data, targets in test_loader:
        data = data.cuda()
        data = Variable(data, requires_grad=True)
        batch_output = model(data)
        soft_out = F.softmax(batch_output, dim=1)
        soft_out_max, _ = torch.max(soft_out.data, dim=1)

        for i in range(data.size(0)):
            MSP_results.append(soft_out_max[i].data.cpu().numpy())

    MSP = np.mean(MSP_results)
    return MSP


def get_upper_lower(model, train_loader, dataset):  # prepare for NBC and SNAC
    neuron_num = 0
    model.eval()
    # temp_x = torch.rand(2, 3, 32, 32).cuda()
    if dataset == 'mnist' or dataset == 'fashion':
        temp_x = torch.rand(2, 1, 28, 28).cuda()
    elif dataset == 'cifar10' or dataset == 'svhn':
        temp_x = torch.rand(2, 3, 32, 32).cuda()
    else:
        print("Don't know the shape of input dataset.")
        temp_x = None
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    upper_value_list, lower_value_list = [], []
    for i in range(num_output):
        neuron_count = temp_list[i].size(1)
        neuron_num += neuron_count
        tmp_up = torch.zeros(size=(neuron_count,))
        tmp_low = torch.full([neuron_count], 100.0)
        upper_value_list.append(tmp_up)
        lower_value_list.append(tmp_low)

    for data, target in train_loader:
        data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
            output, out_features = model.feature_list(data)

        num_output = len(out_features)
        # print("len:", num_output)

        # get hidden features
        for i in range(num_output):
            val_max, index_max = torch.max(out_features[i].data, 0)
            val_min, index_min = torch.min(out_features[i].data, 0)
            # print(i, "-->:", val_min)
            for j in range(len(val_max)):
                # print("1:", lower_value_list[i][j], "2:", val_min[j])
                upper_value_list[i][j] = max(upper_value_list[i][j], val_max[j])
                lower_value_list[i][j] = min(lower_value_list[i][j], val_min[j])

    return upper_value_list, lower_value_list, neuron_num


def calc_tknc(model, test_loader, dataset):
    top1_count, neuron_num = 0, 0
    model.eval()
    if dataset == 'mnist' or dataset == 'fashion':
        temp_x = torch.rand(2, 1, 28, 28).cuda()
    elif dataset == 'cifar10' or dataset == 'svhn':
        temp_x = torch.rand(2, 3, 32, 32).cuda()
    else:
        print("Don't know the shape of input dataset.")
        temp_x = None
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    index_top1 = dict()
    for i in range(num_output):
        neuron_count = temp_list[i].size(1)
        neuron_num += neuron_count
        index_top1[i] = []

    for data, target in test_loader:
        data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
            output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            val_max, index_max = torch.max(out_features[i].data, 1)  # 返回每一行的最大值，且返回索引（返回最大元素在各行的列索引）
            # print("index_max:", set(index_max.tolist()))
            index_top1[i].extend(index_max.tolist())

    for i in range(num_output):
        tmp = set(index_top1[i])
        top1_count += len(tmp)

    return top1_count / neuron_num


def calc_nc(model, test_loader, upper, lower, nc_k, dataset):
    upper_act_id, lower_act_id, act_id = [], [], []
    upper_act_cnt, lower_act_cnt, act_cnt = 0, 0, 0
    model.eval()
    # temp_x = torch.rand(2, 3, 32, 32).cuda()
    if dataset == 'mnist' or dataset == 'fashion':
        temp_x = torch.rand(2, 1, 28, 28).cuda()
    elif dataset == 'cifar10' or dataset == 'svhn':
        temp_x = torch.rand(2, 3, 32, 32).cuda()
    else:
        print("Don't know the shape of input dataset.")
        temp_x = None
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    for data, target in test_loader:
        data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
            output, out_features = model.feature_list(data)

        for i in range(num_output):
            upper_act_id.append([])
            lower_act_id.append([])
            act_id.append([])
            neurons_num = out_features[i].size(1)
            for k in range(neurons_num):
                tmp = out_features[i][:, k]
                if torch.nonzero(tmp > upper[i][k]).numel():
                    upper_act_id[i].append(k)
                if torch.nonzero(tmp < lower[i][k]).numel():
                    lower_act_id[i].append(k)
                if torch.nonzero(tmp > nc_k).numel():
                    act_id[i].append(k)

    for i in range(num_output):
        upper_act_cnt += len(set(upper_act_id[i]))
        lower_act_cnt += len(set(lower_act_id[i]))
        act_cnt += len(set(act_id[i]))

    # print(upper_act_cnt, lower_act_cnt)
    return upper_act_cnt, lower_act_cnt, act_cnt


def calc_lrd(model, train_loader, dataset, batch, trans, dataroot, aug_train_loader, feature_save_path,
                label_save_path, cluster_num):
    # set information about feature extaction
    feature_list, num_output = get_information(model, dataset)
    feature_size = feature_list[-1]
    getLastFeatureCluster(model, train_loader, feature_save_path, label_save_path, cluster_num, feature_size)
    c_cluster = np.load(label_save_path)
    train_loader_c, test_loader_c = My_data_loader.getTargetDataSet(dataset, batch, trans, dataroot, c_cluster)
    sample_mean, precision = lib.sample_estimator(model, cluster_num, feature_list, train_loader_c)

    M_in_train = lib.get_lrd_score(model, train_loader, cluster_num, True, sample_mean, precision, num_output - 1)
    M_in_train = np.asarray(M_in_train, dtype=np.float32)
    Mahalanobis_in_train = M_in_train.reshape((M_in_train.shape[0], -1))
    Mahalanobis_in_train = np.asarray(Mahalanobis_in_train, dtype=np.float32)
    Mahalanobis_in_train = np.array(Mahalanobis_in_train).flatten()  # score(T_train)

    M_in_train_aug = lib.get_lrd_score(model, aug_train_loader, cluster_num, True, sample_mean, precision,
                                          num_output - 1)
    M_in_train_aug = np.asarray(M_in_train_aug, dtype=np.float32)
    Mahalanobis_in_train_aug = M_in_train_aug.reshape((M_in_train_aug.shape[0], -1))
    Mahalanobis_in_train_aug = np.asarray(Mahalanobis_in_train_aug, dtype=np.float32)
    Mahalanobis_in_train_aug = np.array(Mahalanobis_in_train_aug).flatten()  # score(T_train')

    out_test_loader = My_data_loader.get_OOD_Dataset(dataset, batch, trans, dataroot)
    M_out = lib.get_lrd_score(model, out_test_loader, cluster_num, False, sample_mean, precision, num_output - 1)
    M_out = np.asarray(M_out, dtype=np.float32)
    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
    Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
    Mahalanobis_out = np.array(Mahalanobis_out).flatten()  # score(OOD)

    dis1 = wasserstein_distance(Mahalanobis_in_train, Mahalanobis_in_train_aug)
    dis2 = wasserstein_distance(Mahalanobis_in_train, Mahalanobis_out)
    lrd_gen = 1 - (dis1 / dis2)
    return lrd_gen


def getLastFeatureCluster(model, train_loader, feature_save_path, label_save_path, cluster_classes, feature_size):
    if not os.path.exists(feature_save_path):
        Feature = np.zeros(shape=(1, int(feature_size)))
        for data, target in train_loader:
            data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output, out_features = model.feature_list(data)
            last_feature = out_features[-1]
            Feature = np.concatenate((Feature, last_feature.cpu()), axis=0)
        np.save(feature_save_path, Feature[1:])
    # cluster
    X = np.load(feature_save_path)
    # print("saved_feature shape: ", X.shape)
    X_tsne = TSNE(n_components=2).fit_transform(X)
    c_cluster = GMM(n_components=cluster_classes).fit_predict(X_tsne)
    np.save(label_save_path, c_cluster)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_cluster)
    # plt.show()


def get_information(model, dataset):
    model.eval()
    if dataset == 'cifar10' or dataset == 'svhn':
        temp_x = torch.rand(1, 3, 32, 32).cuda()
    elif dataset == 'mnist' or dataset == 'fashion':
        temp_x = torch.rand(1, 1, 28, 28).cuda()
    else:
        temp_x = None
        print("Don't know the size of this dataset.")
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)  # 5
    feature_list = np.empty(num_output)  # feature_list: [64, 64, 128, 256, 512]
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    return feature_list, num_output


if __name__ == '__main__':
    params_map = {
        'rotation_range': (-10, 10),
        'brightness_range': (0.8, 1.3),
        'contrast_range': (0.8, 1.3),
        'shift_range': (0.05, 0.05)
    }

    aug_transform = transforms.Compose([
        transforms.RandomRotation(degrees=params_map['rotation_range']),
        transforms.ColorJitter(brightness=params_map['brightness_range'], contrast=params_map['contrast_range']),
        transforms.RandomAffine(degrees=0, translate=params_map['shift_range']),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    sampler1 = list(range(10000, 20000))
    sampler2 = list(range(0, 100000))
    train_loader, aug_test_loader = My_data_loader.getDataSet_with_aug(data_type="cifar10", dataroot="./data",
                                                                       batch_size=200, ori_TF=trans, test_with_aug=True,
                                                                       aug_TF=aug_transform, sampler=sampler1)
    aug_train_loader, test_loader = My_data_loader.getDataSet_with_aug(data_type="cifar10", dataroot="./data",
                                                                       batch_size=200, ori_TF=trans,
                                                                       train_with_aug=True,
                                                                       aug_TF=aug_transform, sampler=sampler2)
    print("Len of train loader:", len(train_loader), ":Len of aug test loader:", len(aug_test_loader))

    model1 = models.ResNet34(num_c=10)
    model1 = model1.to("cuda")
    cudnn.benchmark = True
    pre_trained_net = "./pre_trained/resnet34_cifar10.pth"  # ori model
    model1.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(0))['net'])
    model1.cuda()

    # act_cnt = calc_avg_nac(model1, test_loader)
    # print(act_cnt)

    # ER1 = cacl_EG(model1, "resnet", aug_test_loader)
    # upper_value_list1, lower_value_list1, neuron_num1 = get_upper_lower(model1, train_loader)
    # upper_act_mix_train1, lower_act_mix_train1 = calc_nbc_snac(model1, aug_train_loader, upper_value_list1, lower_value_list1)
    # NBC_mix_train1 = (upper_act_mix_train1 + lower_act_mix_train1) / (2 * neuron_num1)
    # SNAC_mix_train1 = upper_act_mix_train1/neuron_num1
    # print("mix NBC = ", NBC_mix_train1)
    # print("mix SNAC = ", SNAC_mix_train1)
    # print("Model1 Empirical Generalization (EG) = ", ER1)
