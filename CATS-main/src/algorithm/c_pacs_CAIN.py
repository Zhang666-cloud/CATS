import os
import torch
import torchvision
from torchvision import transforms
from einops import rearrange
from collections import defaultdict
import pickle
from tqdm import tqdm
import random
import numpy as np
from src.model import create_model
from src.options import args_parser


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def calculate_and_save_metrics_pacs(patch_len, device, domain_name, dataset_path):
    args = args_parser()
    model = create_model(args)
    model.change_bn(mode='grad')
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domain_path = os.path.join(dataset_path, domain_name)
    domain_dataset = torchvision.datasets.ImageFolder(root=domain_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=args.batch_size, shuffle=False)

    class_plpd_values = defaultdict(list)
    class_correct = defaultdict(int)  # 存储每个类别的正确预测数
    class_total = defaultdict(int)  # 存储每个类别的总样本数

    with torch.no_grad():
        for X, Y in tqdm(dataloader, total=len(dataloader)):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)
            prob_outputs = logits.softmax(dim=1)

            # 计算准确率
            _, predicted = torch.max(logits, 1)
            correct = (predicted == Y).squeeze()

            # 更新每个类别的准确率统计
            for j in range(len(Y)):
                class_id = Y[j].item()
                class_total[class_id] += 1
                if correct[j].item():
                    class_correct[class_id] += 1

            x_prime = X.clone().detach()
            resize_t = transforms.Resize(
                ((x_prime.shape[-1] // patch_len) * patch_len, (x_prime.shape[-1] // patch_len) * patch_len)
            )
            resize_o = torchvision.transforms.Resize((x_prime.shape[-1], x_prime.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
            x_prime = resize_o(x_prime)
            outputs_prime = model(x_prime)
            prob_outputs = logits.softmax(1)
            prob_outputs_prime = outputs_prime.softmax(1)
            cls1 = prob_outputs.argmax(dim=1)

            plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(
                prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
            plpd = plpd.reshape(-1)

            for j in range(len(Y)):
                class_id = Y[j].item()
                class_plpd_values[class_id].append(plpd[j].item())

    # Calculate and save the average PLPD value for each class
    avg_class_plpd_values = {}
    class_accuracy = {}  # 存储每个类别的准确率
    num_classes = 7  # Assuming PACS has 7 classes per domain

    # 获取类别名称
    class_names = domain_dataset.classes

    for class_id in range(num_classes):
        if class_plpd_values[class_id]:
            avg_class_plpd_values[class_id] = sum(class_plpd_values[class_id]) / len(class_plpd_values[class_id])
        else:
            avg_class_plpd_values[class_id] = 0

        # 计算每个类别的准确率
        if class_total[class_id] > 0:
            class_accuracy[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            class_accuracy[class_id] = 0

    # 保存PLPD值
    metrics_file_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/avg_len_4_plpd_values_PACS1.pkl'
    with open(metrics_file_path, 'wb') as f:
        pickle.dump(avg_class_plpd_values, f)

    # 保存准确率
    accuracy_file_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/accuracy_values_PACS1.pkl'
    with open(accuracy_file_path, 'wb') as f:
        pickle.dump(class_accuracy, f)

    # 打印结果
    print(f"\nResults for domain {domain_name}:")
    for class_id in range(num_classes):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(
            f"Class {class_id} ({class_name}): PLPD = {avg_class_plpd_values[class_id]:.4f}, Accuracy = {class_accuracy[class_id] * 100:.2f}%")


def main():
    set_random_seed(42)
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    patch_len = 4
    dataset_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/data/PACS'

    # Calculate for the first domain 'art_painting'
    calculate_and_save_metrics_pacs(patch_len, device, 'cartoon', dataset_path)


if __name__ == '__main__':
    main()