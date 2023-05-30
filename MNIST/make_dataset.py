import argparse
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import make_folder
# from load_svhn import load_svhn
# from load_cifar10 import load_cifar10
from load_mnist import load_mnist
from glob import glob
from PIL import Image
from tqdm import tqdm

#この関数は、各バッグ（サンプルの集合）に含まれる各クラスのインスタンスの比率をランダムに生成します。
def get_label_proportion(num_bags=100, num_classes=10):
    proportion = np.random.rand(num_bags, num_classes)
    proportion /= proportion.sum(axis=1, keepdims=True)

    return proportion


# この関数が返すNは、各バッグにおける各クラスのインスタンスの数を表す2次元配列です。
# この配列の形状は(num_bags, num_classes)で、各要素は該当するバッグにおける該当するクラスのインスタンスの数を表します。
def get_N_label_proportion(proportion, num_instances, num_classes):
    N = np.zeros(proportion.shape)
    for i in range(len(proportion)):
        p = proportion[i]
        for c in range(len(p)):
            if (c+1) != num_classes:
                num_c = int(np.round(num_instances*p[c]))
                if sum(N[i])+num_c >= num_instances:
                    num_c = int(num_instances-sum(N[i]))
            else:
                num_c = int(num_instances-sum(N[i]))

            N[i][c] = int(num_c)
        np.random.shuffle(N[i])
    print(N.sum(axis=0))
    print((N.sum(axis=1) != num_instances).sum())
    return N 


# def create_bags(data, label, num_posi_bags, num_nega_bags, args):
#     # make poroportion
#     proportion = get_label_proportion(num_posi_bags, args.num_classes)
#     proportion_N = get_N_label_proportion(
#         proportion, args.num_instances, args.num_classes)

#     proportion_N_nega = np.zeros((num_nega_bags, args.num_classes))
#     proportion_N_nega[:, 0] = args.num_instances

#     proportion_N = np.concatenate([proportion_N, proportion_N_nega], axis=0)

#     # make index
#     idx = np.arange(len(label))
#     idx_c = []
#     for c in range(args.num_classes):
#         x = idx[label[idx] == c]
#         np.random.shuffle(x)
#         idx_c.append(x)

#     bags_idx = []
#     for n in range(len(proportion_N)):
#         bag_idx = []
#         for c in range(args.num_classes):
#             sample_c_index = np.random.choice(
#                 idx_c[c], size=int(proportion_N[n][c]), replace=False)
#             bag_idx.extend(sample_c_index)

#         np.random.shuffle(bag_idx)
#         bags_idx.append(bag_idx)
#     # bags_index.shape => (num_bags, num_instances)

#     # make data, label, proportion
#     bags, labels = data[bags_idx], label[bags_idx]
#     original_lps = proportion_N / args.num_instances

#     partial_lps = original_lps.copy()
#     posi_nega = (original_lps[:, 0] != 1)
#     partial_lps[posi_nega == 1, 0] = 0  # mask negative class
#     partial_lps /= partial_lps.sum(axis=1, keepdims=True)  # normalize

#     return bags, labels, original_lps, partial_lps


def create_bags(data, label, num_bags, args):
    # make proportion
    proportion = get_label_proportion(num_bags, args.num_classes)
    proportion_N = get_N_label_proportion(proportion, args.num_instances, args.num_classes)

    # make index
    idx = np.arange(len(label))  # 0, 1, 2, ..., len(label)-1
    idx_c = []
    for c in range(args.num_classes):
        x = idx[label[idx] == c]
        np.random.shuffle(x)
        idx_c.append(x)

    bags_idx = []
    for n in range(len(proportion_N)):
        bag_idx = []
        for c in range(args.num_classes):
            sample_c_index = np.random.choice(idx_c[c], size=int(proportion_N[n][c]), replace=False)
            bag_idx.extend(sample_c_index)

        np.random.shuffle(bag_idx)
        bags_idx.append(bag_idx)

    # make data, label, proportion
    bags, labels = data[bags_idx], label[bags_idx]
    original_lps = proportion_N / args.num_instances

    return bags, labels, original_lps



def main(args):
    # load dataset
    if args.dataset == 'mnist':
        data, label, test_data, test_label = load_mnist()
    # elif args.dataset == 'svhn':
    #     data, label, test_data, test_label = load_svhn()
    # elif args.dataset == 'cifar10':
    #     data, label, test_data, test_label = load_cifar10()

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(data, label)):
        train_data, train_label = data[train_idx], label[train_idx]
        val_data, val_label = data[val_idx], label[val_idx]

        output_path = 'data/%s/%dclass/%d/' % (
            args.dataset, args.num_classes, i)
        make_folder(output_path)

        # train
        bags, labels, original_lps = create_bags(train_data, train_label,
                                                              args.train_num_bags, 
                                                              args)
        np.save('%s/train_bags' % (output_path), bags)
        np.save('%s/train_labels' % (output_path), labels)
        np.save('%s/train_original_lps' % (output_path), original_lps)

        # val
        bags, labels, original_lps = create_bags(val_data, val_label,
                                                              args.val_num_bags, 
                                                              args)
        np.save('%s/val_bags' % (output_path), bags)
        np.save('%s/val_labels' % (output_path), labels)
        np.save('%s/val_original_lps' % (output_path), original_lps)

        # test

        used_test_data, used_test_label = [], []
        for c in range(args.num_classes):
            used_test_data.extend(test_data[test_label == c])
            used_test_label.extend(test_label[test_label == c])
        test_data, test_label = np.array(
            used_test_data), np.array(used_test_label)

        bags, labels, original_lps = create_bags(test_data, test_label,
                                                              args.test_num_bags, 
                                                              args)
        np.save('%s/test_bags' % (output_path), bags)
        np.save('%s/test_labels' % (output_path), labels)
        np.save('%s/test_original_lps' % (output_path), original_lps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_instances', default=32, type=int)

    parser.add_argument('--train_num_bags', default=48, type=int)
    parser.add_argument('--val_num_bags', default=12, type=int)
    parser.add_argument('--test_num_bags', default=10, type=int)

    args = parser.parse_args()

    #################
    np.random.seed(args.seed)
    main(args)
