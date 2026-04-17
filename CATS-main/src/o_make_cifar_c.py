import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os

from .distortions import distortions, test_distortions


def make_cifar10_c(partition_idxs, is_train, data_dir='../data', mode="none"):
    """
    Generate TensorDataset of CIFAR-10-C
        partition_idxs: dictionary of (client index : list of sample indices)
        is_train: whether each client is source client or target client
        data_dir: root of torchvision dataset
        mode: 'iid' or 'ood'.
            'iid' uses the same 15 distortions for both source and target clients
            'ood' uses 15 distortions for source client and 4 additional distortions for target clients
    """
    data_dir = os.path.join(data_dir, 'torchvision')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # no normalization
    ])

    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # this is important
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean and std of each channel
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train_dataset, test_dataset])

    cifar_c, labels = [None] * len(dataset), [None] * len(dataset)

    corruption = {}

    for cid, sids in tqdm(partition_idxs.items()):
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)
        corruption[cid] = (distortion.__name__, severity)
        for sid in sids:
            x, y = dataset[sid]
            x = distortion(x, severity=severity)  # add distortion
            x = np.uint8(x)  # convert back to original space
            x = post_transform(x)
            cifar_c[sid] = x
            labels[sid] = y
    assert None not in cifar_c
    assert None not in labels

    cifar_c = torch.stack(cifar_c)
    labels = torch.LongTensor(labels)

    return cifar_c, labels, corruption


CIFAR100_C_DIR = '/mnt/sda/PythonProject/Learning_Dataset/CIFAR100-C/CIFAR-100-C'


def make_cifar100_c(partition_idxs, is_train, data_dir='../data'):
    """
    Generate TensorDataset of CIFAR-100-C.
    Training客户端分配CIFAR-100数据，测试客户端分配CIFAR-100-C数据。
    """
    # CIFAR-100 数据目录
    data_dir = os.path.join(data_dir, 'torchvision')

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # No normalization
    ])

    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # This is important
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),  # Mean and std of each channel
    ])

    # 加载 CIFAR-100 训练数据
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)

    # 初始化容器
    cifar_c = [None] * len(train_dataset)  # CIFAR100-C 的数据
    labels = [None] * len(train_dataset)  # 标签
    corruption = {}  # 存储腐蚀信息

    # 确保 CIFAR100-C 数据存在
    if not os.path.exists(CIFAR100_C_DIR):
        raise FileNotFoundError(f"CIFAR100-C data not found at {CIFAR100_C_DIR}")

    # 遍历每个客户端
    for cid, sids in tqdm(partition_idxs.items()):
        if is_train[cid]:  # 训练客户端
            for sid in sids:  # 客户端分配的样本索引
                if sid >= len(train_dataset):  # 检查索引是否超出范围
                    raise ValueError(f"Training client index {sid} is out of bounds for CIFAR-100 dataset")

                # 从训练数据集中提取样本
                x, y = train_dataset[sid]
                x = transforms.ToTensor()(x)  # 转换为张量
                x = post_transform(x)  # 归一化
                cifar_c[sid] = x
                labels[sid] = y
            corruption[cid] = "CIFAR-100 training data"

        else:  # 测试客户端
            # 遍历 CIFAR100-C 文件，加载腐蚀数据
            for corruption_file in os.listdir(CIFAR100_C_DIR):
                if not corruption_file.endswith(".npy"):  # 仅处理 .npy 文件
                    continue

                # 完整路径
                corruption_path = os.path.join(CIFAR100_C_DIR, corruption_file)

                # 加载腐蚀数据
                corrupted_data = np.load(corruption_path)  # 形状: (10000, 32, 32, 3)
                if corruption_file == "labels.npy":  # 处理标签文件
                    corrupted_labels = corrupted_data
                    continue
                else:
                    corrupted_images = corrupted_data

                # 对测试客户端分配样本
                for sid in sids:  # 注意 sid 是客户端 ID
                    if sid >= corrupted_images.shape[0]:  # 检查索引是否超出范围
                        raise ValueError(f"Testing client index {sid} is out of bounds for CIFAR-100-C dataset")

                    # 从 CIFAR100-C 数据中提取样本
                    img = corrupted_images[sid]
                    lbl = corrupted_labels[sid]

                    # 转换并归一化图片
                    img = transforms.ToTensor()(img)
                    img = post_transform(img)

                    # 存储图片和标签
                    cifar_c[sid] = img
                    labels[sid] = lbl
            corruption[cid] = "CIFAR100-C test data"

    # 转换为 Tensor
    cifar_c = torch.stack([x for x in cifar_c if x is not None])
    labels = torch.LongTensor([y for y in labels if y is not None])

    return cifar_c, labels, corruption


def random_distortion(range_severity=(1, 6), mode='train'):
    if mode == "train" or mode == "iid":
        selected_distortions = distortions
    elif mode == "ood":
        selected_distortions = test_distortions
    else:
        raise NotImplementedError

    num_distortions = len(selected_distortions)
    i = np.random.randint(num_distortions)  # randomly choose a distortion
    s = np.random.randint(low=range_severity[0], high=range_severity[1])  # randomly choose severity from 1 to 5
    return selected_distortions[i], s


def test():
    partition_idxs = {
        0: [*range(1)],
        1: [*range(1, 2)]
    }
    make_cifar10_c(partition_idxs)
