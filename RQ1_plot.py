import os

from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    file_path = "./rq1_results/rq1.xlsx"
    os.makedirs("./rq1_results/Acc", exist_ok=True)
    os.makedirs("./rq1_results/Loss", exist_ok=True)
    # model_li = ['mnist_resnet18', 'mnist_vgg11', 'fashion_resnet18', 'fashion_vgg11',
    #             'cifar10_densenet', 'cifar10_vgg16', 'svhn_densenet']  # 'svhn_vgg16'
    model_li = ['mnist_resnet18']
    baselines = ['NBC', 'SNAC', 'NAC', 'TKNC', 'LSC', 'LDR']
    plt.rcParams['pdf.use14corefonts'] = True
    for model in model_li:
        data = pd.read_excel(io=file_path, sheet_name=model)
        for baseline in baselines:
            sns.jointplot(x=baseline, y='Acc', data=data, kind='reg', color = 'r')
            plt.tight_layout()
            plt.savefig(f'./rq1_results/Acc/rq1_{model}_{baseline}_Acc.pdf', dpi=200)
            plt.show()
            plt.clf()

            sns.jointplot(x=baseline, y='Loss', data=data, kind='reg', color='mediumblue')
            plt.tight_layout()
            plt.savefig(f'./rq1_results/Loss/rq1_{model}_{baseline}_Loss.pdf', dpi=200)
            plt.show()
            plt.clf()


    # file_path = "./rq2_results/rq2.xlsx"
    # model_li = ['mnist_resnet18', 'mnist_vgg11', 'fashion_resnet18', 'fashion_vgg11',
    #             'cifar10_densenet', 'cifar10_vgg16', 'svhn_densenet']  # 'svhn_vgg16'
    # # model_li = ['mnist_resnet18']
    # baselines = ['NBC', 'SNAC', 'NAC', 'TKNC', 'GENTLE']
    # plt.rcParams['pdf.use14corefonts'] = True
    #
    # for model in model_li:
    #     data = pd.read_excel(io=file_path, sheet_name=model)
    #     for baseline in baselines:
    #         sns.jointplot(x=baseline, y='EG', data=data, kind='reg')
    #         plt.tight_layout()
    #         plt.savefig(f'./rq1_results/fig/rq2_{model}_{baseline}.pdf', dpi=200)
    #         # plt.show()
    #         plt.clf()


def fashion_and_mnist():
    file_path = "./rq2_results/rq2.xlsx"
    baselines = ['NBC', 'SNAC', 'NAC', 'TKNC', 'GENTLE']
    plt.rcParams['pdf.use14corefonts'] = True
    dataset_li = ['mnist', 'fashion']
    for dataset in dataset_li:
        data1 = pd.read_excel(io=file_path, sheet_name=f'{dataset}_vgg11')
        data2 = pd.read_excel(io=file_path, sheet_name=f'{dataset}_resnet18')
        data = data1.append(data2, ignore_index=True)
        for baseline in baselines:
            sns.jointplot(x=baseline, y='EG', data=data, kind='reg')
            plt.tight_layout()
            plt.savefig(f'./rq2_results/fig/rq2_{dataset}_{baseline}.pdf', dpi=200)
            plt.clf()


def cifar10_and_svhn():
    file_path = "./rq2_results/rq2.xlsx"
    baselines = ['NBC', 'SNAC', 'NAC', 'TKNC', 'GENTLE']
    plt.rcParams['pdf.use14corefonts'] = True
    dataset_li = ['cifar10']  # 'svhn'
    for dataset in dataset_li:
        data1 = pd.read_excel(io=file_path, sheet_name=f'{dataset}_vgg16')
        data2 = pd.read_excel(io=file_path, sheet_name=f'{dataset}_densenet')
        data = data1.append(data2, ignore_index=True)
        for baseline in baselines:
            sns.jointplot(x=baseline, y='EG', data=data, kind='reg')
            plt.tight_layout()
            plt.savefig(f'./rq2_results/fig/rq2_{dataset}_{baseline}.pdf', dpi=200)
            plt.clf()
