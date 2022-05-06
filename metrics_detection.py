'''
calculate the OOD detection ability of maxp, ODIN, NBC, SNAC
'''
import lib
import My_data_loader
import calculate_log as callog
import metrics
import os
import models
import torch.backends.cudnn as cudnn
import torch
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegressionCV


def get_performance(model, net_type, dataset, outf, batch, dataroot, test_loader, out_dist_list, trans):
    """ measure the performance of MaxP, ODIN, OE, Energy """
    T_list = [1, 10, 100, 1000]
    maxp_line_list, oe_line_list = [], []
    ODIN_best_tnr = [0] * len(out_dist_list)
    ODIN_best_results = [0] * len(out_dist_list)
    ODIN_best_temperature = [-1] * len(out_dist_list)

    energy_best_tnr = [0] * len(out_dist_list)
    energy_best_results = [0] * len(out_dist_list)
    energy_best_temperature = [-1] * len(out_dist_list)

    for T in T_list:
        temperature, magnitude = T, 0
        lib.get_posterior(model, net_type, test_loader, magnitude, temperature, outf, "ID", dataset)
        out_count = 0
        print('Temperature: ' + str(temperature) + ' / Noise: ' + str(magnitude))
        for out_dist in out_dist_list:
            out_test_loader = My_data_loader.getNonTargetDataSet(out_dist, batch, trans, dataroot)
            print('Out-distribution: ' + out_dist)
            print("Len:", len(out_test_loader))
            lib.get_posterior(model, net_type, out_test_loader, magnitude, temperature, outf, "OOD", out_dist)
            if temperature == 1 and magnitude == 0:  # maxp, NBC, SNAC 的指标
                test_results_maxp = callog.metric(outf, ['Test'], file_title="confidence", in_dataset_name=dataset, out_dataset_name=out_dist, sort_reverse=False)
                test_results_oe = callog.metric(outf, ['Test'], file_title="OE", in_dataset_name=dataset, out_dataset_name=out_dist, sort_reverse=False)
                maxp_line_list.append(test_results_maxp)
                oe_line_list.append(test_results_oe)
            else:  # ODIN 的指标
                val_results = callog.metric(outf, ['Val'], file_title="confidence", in_dataset_name=dataset, out_dataset_name=out_dist, sort_reverse=False)
                energy_val_results = callog.metric(outf, ['Val'], file_title="energy", in_dataset_name=dataset, out_dataset_name=out_dist, sort_reverse=False)
                if ODIN_best_tnr[out_count] < val_results['Val']['TNR']:
                    ODIN_best_tnr[out_count] = val_results['Val']['TNR']
                    ODIN_best_results[out_count] = callog.metric(outf, ['Test'], file_title="confidence", in_dataset_name=dataset, out_dataset_name=out_dist, sort_reverse=False)
                    ODIN_best_temperature[out_count] = temperature
                if energy_best_tnr[out_count] < energy_val_results['Val']['TNR']:
                    energy_best_tnr[out_count] = energy_val_results['Val']['TNR']
                    energy_best_results[out_count] = callog.metric(outf, ['Test'], file_title="energy", in_dataset_name=dataset, out_dataset_name=out_dist, sort_reverse=False)
                    energy_best_temperature[out_count] = temperature
            out_count += 1

    return maxp_line_list, oe_line_list, ODIN_best_results, ODIN_best_temperature, energy_best_results, energy_best_temperature


def get_gentle_performance(model, net_type, dataset, outf, batch, dataroot, train_loader, test_loader, out_dist_list, trans, cluster_num_list):
    Gentle_best_tnr = [0] * len(out_dist_list)
    Gentle_best_results = [0] * len(out_dist_list)
    Gentle_best_cluster = [-1] * len(out_dist_list)

    feature_save_path = outf + 'feature_' + net_type + '_' + dataset + '.npy'

    for cluster_num in cluster_num_list:
        print('Cluster number: ' + str(cluster_num))

        # set information about feature extaction
        feature_list, num_output = metrics.get_information(model, dataset)
        label_save_path = outf + 'label_' + net_type + '_' + dataset + '_' + str(cluster_num) + 'classes.npy'
        feature_size = feature_list[-1]
        metrics.getLastFeatureCluster(model, train_loader, feature_save_path, label_save_path, cluster_num, feature_size)
        c_cluster = np.load(label_save_path)
        train_loader_c, test_loader_c = My_data_loader.getTargetDataSet(dataset, batch, trans, dataroot, c_cluster)

        # prepare for Gentle
        sample_mean, precision = lib.sample_estimator(model, cluster_num, feature_list, train_loader_c)
        print("sample_mean[-1] shape:\n", sample_mean[-1].shape, ". len of sample_mean:", len(sample_mean))
        print("precision[-1] shape:\n", precision[-1].shape, ". len of precision:", len(precision))

        print('Geting Gentle scores')
        lib.get_gentle_score(model, test_loader, cluster_num, "ID", sample_mean, precision, num_output - 1,
                             write_file=True, dataset_name=dataset, outf=outf)

        out_count = 0

        for out_dist in out_dist_list:
            out_test_loader = My_data_loader.getNonTargetDataSet(out_dist, batch, trans, dataroot)
            print('Out-distribution: ' + out_dist)
            lib.get_gentle_score(model, out_test_loader, cluster_num, "OOD", sample_mean, precision,
                                 num_output - 1, write_file=True, dataset_name=out_dist, outf=outf)
            val_results = callog.metric(outf, ['Val'], file_title='Gentle', in_dataset_name=dataset,
                                        out_dataset_name=out_dist, sort_reverse=True)
            # print(val_results)
            if Gentle_best_tnr[out_count] < val_results['Val']['TNR']:
                Gentle_best_tnr[out_count] = val_results['Val']['TNR']
                Gentle_best_results[out_count] = callog.metric(outf, ['Test'], file_title='Gentle',
                                                               in_dataset_name=dataset, out_dataset_name=out_dist,
                                                               sort_reverse=True)
                Gentle_best_cluster[out_count] = cluster_num

            out_count += 1
    return Gentle_best_results, Gentle_best_cluster


def show_detection_score(best_results, out_dist_list, best_temperature=None, best_cluster=None):
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT', 'FPR']
    count_out = 0
    for results in best_results:
        print('out_distribution: ' + out_dist_list[count_out])
        for mtype in mtypes:
            if mtype == "FPR":
                print(' {mtype:6s}'.format(mtype=mtype + '(↓) '), end='')
            else:
                print(' {mtype:6s}'.format(mtype=mtype + '(↑) '), end='')
        print('\n{val:6.2f}'.format(val=100. * results['Test']['TNR']), end='')
        print(' {val:6.2f}'.format(val=100. * results['Test']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100. * results['Test']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100. * results['Test']['AUIN']), end='')
        print(' {val:6.2f}'.format(val=100. * results['Test']['AUOUT']), end='')
        print(' {val:6.2f}\n'.format(val=100. * results['Test']['FPR']), end='')
        if best_temperature is not None:
            print('temperature: ' + str(best_temperature[count_out]))
        if best_cluster is not None:
            print('cluster number: ' + str(best_cluster[count_out]))
        print('')
        count_out += 1


def save_information_mahala(model, dataset, outf, batch, dataroot, train_loader, test_loader, out_dist_list, trans, num_classes):
    # set information about feature extaction
    feature_list, num_output = metrics.get_information(model, dataset)

    # prepare for Mahalanobis
    Maha_sample_mean, Maha_precision = lib.sample_estimator(model, num_classes, feature_list, train_loader)
    print("sample_mean:\n", Maha_sample_mean[0].shape, len(Maha_sample_mean))
    print("precision:\n", Maha_precision[0].shape, len(Maha_precision))

    print('Geting Mahalanobis scores')
    magnitude = 0.0
    print('Noise: ' + str(magnitude))

    M_in = lib.get_gentle_score(model, test_loader, num_classes, "ID", Maha_sample_mean, Maha_precision, num_output-1)
    M_in = np.asarray(M_in, dtype=np.float32)
    Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))

    for out_dist in out_dist_list:
        out_test_loader = My_data_loader.getNonTargetDataSet(out_dist, batch, trans, dataroot)
        print('Out-distribution: ' + out_dist)
        M_out = lib.get_gentle_score(model, out_test_loader, num_classes, "OOD", Maha_sample_mean, Maha_precision, num_output-1)
        M_out = np.asarray(M_out, dtype=np.float32)
        Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))

        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_data, Mahalanobis_labels = lib.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
        file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), dataset, out_dist))
        Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(file_name, Mahalanobis_data)


def get_mahala_preformance(dataset, outf, out_dist_list):
    list_results = []
    for out in out_dist_list:
        print('Out-of-distribution: ', out)
        score = 'Mahalanobis_0.0'
        total_X, total_Y = lib.load_characteristics(score, dataset, out, outf)
        X_val, Y_val, X_test, Y_test = lib.block_split(total_X, total_Y, out)
        X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
        Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
        X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
        Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
        results = lib.detection_performance(lr, X_val_for_test, Y_val_for_test, outf, dataset, out)
        list_results.append(results)
    return list_results
