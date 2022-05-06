from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import calculate_log as callog
from torch.autograd import Variable
import metrics
from scipy.spatial.distance import pdist, cdist, squareform


def get_gaussian_score(model, data, layer_index, num_classes, sample_mean, precision):
    data = data.cuda()
    data = Variable(data, requires_grad=True)

    output, out_features_list = model.feature_list(data)
    out_features = out_features_list[layer_index]

    # compute Gentle score
    gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

    gaussian_score, _ = torch.max(gaussian_score, dim=1)
    return gaussian_score


def calc_cov(model, test_data, neuron_num, upper, lower, nc_k=0.75):  # 计算单个测试用例的 NBC, SNAC, NAC
    nbc_list, snac_list, nac_list = [], [], []
    model.eval()
    data = test_data.cuda()
    with torch.no_grad():
        data = Variable(data)
        output, out_features = model.feature_list(data)
    num_output = len(out_features)  # layer numbers
    upper_cnt = [0] * data.size(0)
    lower_cnt = [0] * data.size(0)
    act_cnt = [0] * data.size(0)

    # get hidden features
    for i in range(num_output):  # DNN的层数
        neurons_num = out_features[i].size(1)

        for j in range(len(out_features[i])):  # 第 j 个测试用例
            tmp_test = out_features[i][j]
            for k in range(neurons_num):  # 第 i 层的第 k 个神经元输出
                tmp = tmp_test[k]
                upper_cnt[j] += len(torch.nonzero(tmp > upper[i][k]))
                lower_cnt[j] += len(torch.nonzero(tmp < lower[i][k]))
                act_cnt[j] += len(torch.nonzero(tmp > nc_k))
    for u, l, a in zip(upper_cnt, lower_cnt, act_cnt):
        # print("NBC=", (u+l)/(2*neuron_num), "SNAC=", u/neuron_num)
        nbc_list.append((u+l)/(2*neuron_num))
        snac_list.append(u/neuron_num)
        nac_list.append(a/neuron_num)
    return nbc_list, snac_list, nac_list


def get_cov(model, test_loader, upper, lower, neuron_num, outf, dataset_name, out_flag):
    NBC_results, SNAC_results, NAC_results = [], [], []
    model.eval()
    total = 0
    temp_file_name_val_NBC = '%s/NBC_Val_%s_%s.txt' % (outf, dataset_name, out_flag)
    temp_file_name_test_NBC = '%s/NBC_Test_%s_%s.txt' % (outf, dataset_name, out_flag)
    temp_file_name_val_SNAC = '%s/SNAC_Val_%s_%s.txt' % (outf, dataset_name, out_flag)
    temp_file_name_test_SNAC = '%s/SNAC_Test_%s_%s.txt' % (outf, dataset_name, out_flag)
    temp_file_name_val_NAC = '%s/SNAC_Val_%s_%s.txt' % (outf, dataset_name, out_flag)
    temp_file_name_test_NAC = '%s/SNAC_Test_%s_%s.txt' % (outf, dataset_name, out_flag)

    if os.path.exists(temp_file_name_val_NBC) and os.path.exists(temp_file_name_test_NBC) and os.path.exists(
            temp_file_name_val_SNAC) and os.path.exists(temp_file_name_test_SNAC):
        return None

    g_NBC = open(temp_file_name_val_NBC, 'w')
    f_NBC = open(temp_file_name_test_NBC, 'w')
    g_SNAC = open(temp_file_name_val_SNAC, 'w')
    f_SNAC = open(temp_file_name_test_SNAC, 'w')
    g_NAC = open(temp_file_name_val_NAC, 'w')
    f_NAC = open(temp_file_name_test_NAC, 'w')

    for data, _ in test_loader:
        total += data.size(0)
        nbc_list, snac_list, nac_list = calc_cov(model, data, neuron_num, upper, lower)
        # print("nbc_list:", nbc_list, "\n snac_list:", snac_list)
        NBC_results.extend(nbc_list)
        SNAC_results.extend(snac_list)
        NAC_results.extend(nac_list)
        # print("NBC_results:", NBC_results, "\n SNAC_results:", SNAC_results)
        # print("len NBC,", len(NBC_results), len(SNAC_results))
        # print(data.size(0))
        for i in range(data.size(0)):
            # print("-->", i, ",", NBC_results[i], SNAC_results[i])
            if total <= 1000:  # val
                g_NBC.write("{}\n".format(NBC_results[i]))
                g_SNAC.write("{}\n".format(SNAC_results[i]))
                g_NAC.write("{}\n".format(NAC_results[i]))
            else:  # test
                f_NBC.write("{}\n".format(NBC_results[i]))
                f_SNAC.write("{}\n".format(SNAC_results[i]))
                f_NAC.write("{}\n".format(NAC_results[i]))

    f_NBC.close()
    f_SNAC.close()
    f_NAC.close()
    g_NBC.close()
    g_SNAC.close()
    g_NAC.close()


def get_gentle_score(model, test_loader, num_classes, out_flag, sample_mean, precision, layer_index,
                     write_file=False, dataset_name=None, outf=None):
    model.eval()
    total = 0

    if write_file:
        temp_file_name_val = '%s/Gentle_Val_%s_%s.txt' % (outf, dataset_name, out_flag)
        temp_file_name_test = '%s/Gentle_Test_%s_%s.txt' % (outf, dataset_name, out_flag)

        g = open(temp_file_name_val, 'w')
        f = open(temp_file_name_test, 'w')

        for data, _ in test_loader:
            total += data.size(0)
            gaussian_score = get_gaussian_score(model, data, layer_index, num_classes, sample_mean, precision)
            for i in range(data.size(0)):
                if total <= 1000:  # val
                    g.write("{}\n".format(gaussian_score[i]))  # noise_gaussian_score
                else:  # test
                    f.write("{}\n".format(gaussian_score[i]))  # noise_gaussian_score
        g.close()
        f.close()
    else:
        score = []
        for data, _ in test_loader:
            gaussian_score = get_gaussian_score(model, data, layer_index, num_classes, sample_mean, precision)
            score.extend(gaussian_score.cpu().numpy())
        return score

def get_gentle_score_without_label(model, test_loader, num_classes, out_flag, sample_mean, precision, layer_index,
                     write_file=False, dataset_name=None, outf=None):
    model.eval()
    total = 0

    if write_file:
        temp_file_name_val = '%s/Gentle_Val_%s_%s.txt' % (outf, dataset_name, out_flag)
        temp_file_name_test = '%s/Gentle_Test_%s_%s.txt' % (outf, dataset_name, out_flag)

        g = open(temp_file_name_val, 'w')
        f = open(temp_file_name_test, 'w')

        for data in test_loader:
            total += data.size(0)
            gaussian_score = get_gaussian_score(model, data, layer_index, num_classes, sample_mean, precision)
            for i in range(data.size(0)):
                if total <= 1000:  # val
                    g.write("{}\n".format(gaussian_score[i]))  # noise_gaussian_score
                else:  # test
                    f.write("{}\n".format(gaussian_score[i]))  # noise_gaussian_score
        g.close()
        f.close()
    else:
        score = []
        for data in test_loader:
            gaussian_score = get_gaussian_score(model, data, layer_index, num_classes, sample_mean, precision)
            score.extend(gaussian_score.cpu().numpy())
        return score

def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    # correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        # total += data.size(0)
        data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
            output, out_features = model.feature_list(data)

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    return sample_class_mean, precision


def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


def get_posterior(model, net_type, test_loader, magnitude, temperature, outf, out_flag, dataset_name):
    '''
    Compute the maximum value of (processed) posterior distribution - ODIN
    return: null
    '''
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    to_np = lambda x: x.data.cpu().numpy()

    temp_file_name_val = '%s/confidence_Val_%s_%s.txt' % (outf, dataset_name, out_flag)
    temp_file_name_test = '%s/confidence_Test_%s_%s.txt' % (outf, dataset_name, out_flag)
    energy_file_name_val = '%s/energy_Val_%s_%s.txt' % (outf, dataset_name, out_flag)
    energy_file_name_test = '%s/energy_Test_%s_%s.txt' % (outf, dataset_name, out_flag)
    oe_file_name_val = '%s/OE_Val_%s_%s.txt' % (outf, dataset_name, out_flag)
    oe_file_name_test = '%s/OE_Test_%s_%s.txt' % (outf, dataset_name, out_flag)

    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    g_energy = open(energy_file_name_val, 'w')
    f_energy = open(energy_file_name_test, 'w')
    g_oe = open(oe_file_name_val, 'w')
    f_oe = open(oe_file_name_test, 'w')

    for data, _ in test_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, requires_grad=True)
        batch_output = model(data)

        # energy score
        energy_score = -to_np((temperature * torch.logsumexp(batch_output / temperature, dim=1)))

        # OE score
        oe_score = to_np((batch_output.mean(1) - torch.logsumexp(batch_output, dim=1)))

        # # temperature scaling
        outputs = batch_output / temperature
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)

        for i in range(data.size(0)):
            if total <= 1000:
                g.write("{}\n".format(1 - soft_out[i]))
                g_energy.write("{}\n".format(energy_score[i]))
                g_oe.write("{}\n".format(oe_score[i]))
            else:
                f.write("{}\n".format(1 - soft_out[i]))
                f_energy.write("{}\n".format(energy_score[i]))
                f_oe.write("{}\n".format(oe_score[i]))

    f.close()
    g.close()
    f_energy.close()
    g_energy.close()
    f_oe.close()
    g_oe.close()


def get_score(model, test_loader, energy_temperature, ODIN_temperature, save_dir, out_flag, dataset_name):
    model.eval()
    to_np = lambda x: x.data.cpu().numpy()
    maxp_path = '%s/maxp_%s_%s.txt' % (save_dir, dataset_name, out_flag)
    oe_path = '%s/oe_%s_%s.txt' % (save_dir, dataset_name, out_flag)
    energy_path = '%s/energy_%s_%s.txt' % (save_dir, dataset_name, out_flag)
    odin_path = '%s/odin_%s_%s.txt' % (save_dir, dataset_name, out_flag)

    maxp_file = open(maxp_path, 'w')
    oe_file = open(oe_path, 'w')
    energy_file = open(energy_path, 'w')
    odin_file = open(odin_path, 'w')

    for data, _ in test_loader:
        data = data.cuda()
        data = Variable(data, requires_grad=True)
        batch_output = model(data)
        energy_score = -to_np((energy_temperature * torch.logsumexp(batch_output / energy_temperature, dim=1)))
        for i in range(data.size(0)):
            energy_file.write("{}\n".format(energy_score[i]))

    for data, _ in test_loader:
        data = data.cuda()
        data = Variable(data, requires_grad=True)
        batch_output = model(data)
        outputs = batch_output / ODIN_temperature
        outputs = outputs / ODIN_temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)
        for i in range(data.size(0)):
            odin_file.write("{}\n".format(1 - soft_out[i]))

    for data, _ in test_loader:
        data = data.cuda()
        data = Variable(data, requires_grad=True)
        batch_output = model(data)
        oe_score = to_np((batch_output.mean(1) - torch.logsumexp(batch_output, dim=1)))
        soft_out = F.softmax(batch_output, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)
        for i in range(data.size(0)):
            maxp_file.write("{}\n".format(1 - soft_out[i]))
            oe_file.write("{}\n".format(oe_score[i]))

    maxp_file.close()
    oe_file.close()
    energy_file.close()
    odin_file.close()


def get_softmax_out(model, test_loader):
    softmax_out = np.zeros((1, 10))
    model.eval()
    for data, _ in test_loader:
        data = data.cuda()
        data = Variable(data, requires_grad=True)
        outputs = model(data)
        soft_out = F.softmax(outputs, dim=1)
        softmax_out = np.concatenate((softmax_out, soft_out.cpu().detach()), axis=0)
    return softmax_out[1:]


def load_characteristics(score, dataset, out, outf, method="Mahala", cluser_num=None):
    """
    Load the calculated scores
    return: data and label of input score
    """
    X, Y, file_name = None, None, None

    if method == "Mahala":
        file_name = os.path.join(outf, "%s_%s_%s.npy" % (score, dataset, out))
    elif method == "Gentle":
        file_name = os.path.join(outf, "%sLabels_%s_%s_%s.npy" % (str(cluser_num), score, dataset, out))
    data = np.load(file_name)

    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1]  # labels only need to load once

    return X, Y


def block_split(X, Y, out):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    if out == 'svhn':
        partition = 26032
    elif out == 'emnist_letters':
        partition = 20800
    else:
        partition = 10000
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: :], Y[partition: :]
    num_train = 1000

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test


def detection_performance(regressor, X, Y, outf, dataset, out_dist):
    """
    Measure the detection performance
    return: detection metrics
    """
    num_samples = X.shape[0]
    l1 = open('%s/Mahalanobis_Test_%s_ID.txt' % (outf, dataset),  'w')
    l2 = open('%s/Mahalanobis_Test_%s_OOD.txt' % (outf, out_dist), 'w')
    y_pred = regressor.predict_proba(X)[:, 1]

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, ['Test'], file_title='Mahalanobis', in_dataset_name=dataset, out_dataset_name=out_dist, sort_reverse=True)
    return results