import argparse
from selection import *
from models.densenet import DenseNet3 as cifar_DenseNet3
from models.vgg import VGG as cifar_VGG
from models_mnist.vgg import VGG as mnist_VGG
from models_mnist.resnet import ResNet18 as mnist_ResNet18
import torch
import os
import metrics
import sys
from datetime import datetime

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RQ3: Selecting data to retrain DNNs.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataset', required=True, help='cifar10 | svhn | mnist | fashion')
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--pre_trained_model', default='./pre_trained', help='path to pre_trained_models')
    parser.add_argument('--data_save', default='./data/save/', help='path to save the to-select dataset')
    parser.add_argument('--net_type', required=True, help='resnet18 | vgg16 | densenet | vgg16')
    parser.add_argument('--outf', default='./output/', help='folder to output results')
    parser.add_argument('--re_epoch', default=10, type=int, help='epochs for retraining')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--lr', type=float, default=0.01, help='gpu index')
    args = parser.parse_args()
    print(args)

    os.makedirs('./print/', exist_ok=True)

    now = datetime.now()
    strtime = now.strftime('%b%d%H%M')
    outfile = './print/' + args.dataset + '_' + args.net_type + '_rq2_' + strtime + '.txt'
    sys.stdout = Logger(outfile)

    dataset, net_type, dataroot, save_dir = args.dataset, args.net_type, args.dataroot, args.data_save
    model, re_model, out_dist_list = None, None, None
    train_num, test_num, num_classes, cluster_num = 50000, 10000, 10, 10
    batch = args.batch_size
    lr = args.lr
    re_epoch = int(args.re_epoch)
    if net_type == 'densenet':
        loader_batch = 100
    else:
        loader_batch = 200

    if dataset == 'svhn':
        cluster_num = 10
        train_num, test_num = 73257, 26032
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        trans = svhn_test_trans
    elif dataset == 'cifar100':
        num_classes = 100
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        trans = cifar_test_trans
    elif dataset == "cifar10":
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        trans = cifar_test_trans
    elif dataset == "mnist":
        train_num = 60000
        out_dist_list = ['emnist_letters', 'kmnist', 'fashion']
        trans = mnist_test_trans
    elif dataset == "fashion":
        train_num = 60000
        out_dist_list = ['mnist', 'kmnist', 'emnist_letters']
        trans = fashion_test_trans

    pre_trained_dir = args.pre_trained_model + "/" + args.net_type + "_" + args.dataset + "/"
    pre_trained_net = pre_trained_dir + "/" + args.net_type + "_" + args.dataset + "_aug0.pth"
    outf = './output/' + dataset + '_' + net_type + '/'
    os.makedirs(outf, exist_ok=True)
    if dataset == 'cifar10' or dataset == 'svhn':
        if net_type == "densenet":
            model = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
            re_model = cifar_DenseNet3(100, num_classes=num_classes).to("cuda")
        elif net_type == "vgg16":
            model = cifar_VGG('VGG16').to("cuda")
            re_model = cifar_VGG('VGG16').to("cuda")
    if dataset == 'fashion' or dataset == 'mnist':
        if net_type == "resnet18":
            model = mnist_ResNet18(num_c=num_classes).to("cuda")
            re_model = mnist_ResNet18(num_c=num_classes).to("cuda")
        elif net_type == "vgg11":
            model = mnist_VGG('VGG11').to("cuda")
            re_model = mnist_VGG('VGG11').to("cuda")

    model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu))['net'])
    model.cuda()

    # step1: 生成并保存待选集， 如果文件已经存在就不会再次生成
    save_data(dataroot, dataset, save_dir)

    # step2: 加载待选集，确定选择比例
    img_path = save_dir + dataset + '_train_aug_ood_image.npy'
    label_path = save_dir + dataset + '_train_aug_ood_label.npy'
    loader, to_select_num = get_to_select_loader(img_path, label_path, dataroot, dataset, loader_batch)
    select_ratio = 0.05
    select_num = int(to_select_num * select_ratio)
    print("select num:", select_num)
    train_loader, test_loader, aug_test_loader, test_and_aug_loader = get_train_and_aug_test(dataroot, dataset,
                                                                                             loader_batch, test_num)
    total_to_select_list = list(range(0, to_select_num))
    baselines = ['kernel', 'least', 'margin', 'entropy', 'random', 'gini', 'mcp', 'lsa']
    # baselines = ['kernel', 'least', 'random', 'gini']


    # step3: 执行 active learning 相关的选择策略
    least_select = least_sampling(loader, model, select_num)
    least_id_in_inloader = [total_to_select_list[i] for i in least_select if total_to_select_list[i] < train_num]
    print("least_select dataset size: ", len(least_id_in_inloader))
    least_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=least_id_in_inloader)

    margin_select = margin_sampling(loader, model, select_num)
    margin_id_in_inloader = [total_to_select_list[i] for i in margin_select if total_to_select_list[i] < train_num]
    print("margin_select dataset size: ", len(margin_id_in_inloader))
    margin_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=margin_id_in_inloader)

    entropy_select = entropy_sampling(loader, model, select_num)
    entropy_id_in_inloader = [total_to_select_list[i] for i in entropy_select if total_to_select_list[i] < train_num]
    print("entropy_select dataset size: ", len(entropy_id_in_inloader))
    entropy_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=entropy_id_in_inloader)

    # step4: 执行 SE 出现的选择策略
    # random_select = random_sampling(to_select_num, select_num) # 规定比例的随机选择
    random_select = random_sampling(to_select_num, to_select_num)  # 全选
    random_id_in_inloader = [total_to_select_list[i] for i in random_select if total_to_select_list[i] < train_num]
    print("random_select dataset size: ", len(random_id_in_inloader))
    random_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=random_id_in_inloader)

    gini_select = gini_sampling(loader, model, net_type, select_num)
    gini_id_in_inloader = [total_to_select_list[i] for i in gini_select if total_to_select_list[i] < train_num]
    print("gini_select dataset size: ", len(gini_id_in_inloader))
    gini_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=gini_id_in_inloader)

    mcp_select = mcp_sampling(loader, model, net_type, select_num)
    mcp_id_in_inloader = [total_to_select_list[i] for i in mcp_select if total_to_select_list[i] < train_num]
    print("mcp_select dataset size: ", len(mcp_id_in_inloader))
    mcp_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=mcp_id_in_inloader)

    lsa_threshold = 0.01
    if dataset == 'mnist' and net_type == 'resnet18':
        lsa_threshold = 0.08
    elif dataset == 'mnist' and net_type == 'vgg11':
        lsa_threshold = 0.008
    elif dataset == 'fashion' and net_type == 'vgg11':
        lsa_threshold = 0.01
    elif dataset == 'fashion' and net_type == 'resnet18':
        lsa_threshold = 0.2
    elif dataset == 'svhn' and net_type == 'vgg16':
        lsa_threshold = 0.003
    elif dataset == 'svhn' and net_type == 'densenet':
        lsa_threshold = 0.03
    elif dataset == 'cifar10' and net_type == 'densenet':
        lsa_threshold = 0.01
    elif dataset == 'cifar10' and net_type == 'vgg16':
        lsa_threshold = 0.00002
    kdes, removed_cols = get_lsa_kdes(model, train_loader, lsa_threshold=lsa_threshold)
    lsa_select = lsa_sampling(loader, model, select_num, kdes, removed_cols)
    lsa_id_in_inloader = [total_to_select_list[i] for i in lsa_select if total_to_select_list[i] < train_num]
    print("lsa_select dataset size: ", len(lsa_id_in_inloader))
    lsa_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=lsa_id_in_inloader)

    # CES 速度太慢，舍弃
    # ces_select = CES_sampling(model, img_path, label_path, dataroot, dataset, loader, to_select_num, select_num)
    # ces_id_in_inloader = [total_to_select_list[i] for i in ces_select if total_to_select_list[i] < train_num]
    # print("ces_select dataset size: ", len(ces_id_in_inloader))
    # ces_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=ces_id_in_inloader)

    # step5: 执行 Gentle 的选择策略
    mean_path = outf + str(cluster_num) + 'classes_mean.npy'
    precision_path = outf + str(cluster_num) + 'classes_precision.npy'
    sample_mean, precision = get_mean_precision(model, net_type, dataset, dataroot, cluster_num, loader_batch,
                                                train_loader, outf, mean_path, precision_path)
    feature_list, num_output = metrics.get_information(model, dataset)
    # lib.get_gentle_score(model, aug_test_loader, cluster_num, "ID", sample_mean, precision, num_output - 1,
    #                      write_file=True, dataset_name=dataset, outf=outf)
    # kernel_in = get_gentle_kernel(outf, dataset)
    # kernel_select = kernel_sampling(loader, model, dataset, cluster_num, sample_mean, precision, kernel_in, select_num)
    # kernel_id_in_inloader = [total_to_select_list[i] for i in kernel_select if total_to_select_list[i] < train_num]
    # print("kernel_select dataset size: ", len(kernel_id_in_inloader))
    # kernel_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch, sampler=kernel_id_in_inloader)
    kernel_select = new_gentle_sampling(loader, train_loader, model, dataset, cluster_num, sample_mean, precision,
                                        select_num)
    kernel_id_in_inloader = [total_to_select_list[i] for i in kernel_select if total_to_select_list[i] < train_num]
    print("kernel_select dataset size: ", len(kernel_id_in_inloader))
    kernel_loader, _ = get_to_select_loader(img_path, label_path, dataroot, dataset, batch,
                                            sampler=kernel_id_in_inloader)

    # step6: 重训练及结果，重训练的验证集是 test + test'
    print("ori model:")
    ori_acc = get_acc(model, aug_test_loader)  # test_and_aug_loader
    for baseline in baselines:
        print("Approach: ", baseline)
        tmp_acc = get_acc(model, eval(baseline + '_loader'))

    for baseline in baselines:
        print("--------", baseline, "--------")
        retrain_dir = args.pre_trained_model + "/" + args.net_type + "_" + args.dataset + "/"
        retrain_path = pre_trained_dir + "/" + net_type + "_" + dataset + '_' + baseline + "_retrain" + str(re_epoch) + ".pth"
        retrain_model(net_type, dataset, num_classes, pre_trained_net, retrain_path, eval(baseline + '_loader'),
                      aug_test_loader, re_epoch, lr)  # # test_and_aug_loader

    # ----------- 验证结果 -------------
    EG_ori, EG_kernel, EG_gentle, EG_least, EG_margin, EG_entropy, EG_random, EG_gini, EG_mcp, EG_lsa = [],[],[],[],[],[],[],[],[],[]
    mix_EG_ori, mix_EG_kernel, mix_EG_gentle, mix_EG_least, mix_EG_margin, mix_EG_entropy, mix_EG_random, mix_EG_gini, mix_EG_mcp, mix_EG_lsa = [],[],[],[],[],[],[],[],[],[]
    ACC_ori, ACC_kernel, ACC_gentle, ACC_least, ACC_margin, ACC_entropy, ACC_random, ACC_gini, ACC_mcp, ACC_lsa = [], [], [], [], [], [], [], [], [], []

    for i in range(5):  # 算 EG 用的是 test'
        train_loader, test_loader, aug_test_loader, test_and_aug_loader = get_train_and_aug_test(dataroot, dataset,
                                                                                                 loader_batch, test_num)
        EG0 = metrics.cacl_EG(model, net_type, aug_test_loader)
        EG_ori.append(EG0)
        ACC = get_acc(model, aug_test_loader)
        ACC_ori.append(ACC)
        print("ori EG0: ", EG0)
        for j, baseline in enumerate(baselines):
            print("--------", baseline, "--------")
            retrain_dir = args.pre_trained_model + "/" + args.net_type + "_" + args.dataset + "/"
            retrain_path = pre_trained_dir + "/" + net_type + "_" + dataset + '_' + baseline + "_retrain" + str(re_epoch) + ".pth"
            re_model.load_state_dict(torch.load(retrain_path, map_location="cuda:" + str(args.gpu))['net'])
            re_model.cuda()
            EG = metrics.cacl_EG(re_model, net_type, aug_test_loader)
            print(EG)
            eval('EG_'+baseline).append(EG)
            eval('ACC_' + baseline).append(get_acc(re_model, aug_test_loader))

    print("EG ori:", np.mean(EG_ori))
    for baseline in baselines:
        print('EG-', baseline, ": ", np.mean(eval('EG_'+baseline)))
        print('deltaEG-', baseline, ": ", np.mean(EG_ori) - np.mean(eval('EG_' + baseline)))
    print("ACC ori:", np.mean(ACC_ori))
    for baseline in baselines:
        print('ACC-', baseline, ": ", np.mean(eval('ACC_' + baseline)))
        print('deltaACC-', baseline, ": ", np.mean(eval('ACC_' + baseline)) - np.mean(ACC_ori))