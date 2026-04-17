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

def calculate_and_save_metrics(patch_len, device):
    args = args_parser()
    model = create_model(args)
    model.change_bn(mode='grad')
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    dataset_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/data/torchvision'
    train_dataset = torchvision.datasets.CIFAR100(
        root=dataset_path, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=dataset_path, train=False, download=True, transform=transform
    )

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    class_plpd_values = defaultdict(list)

    with torch.no_grad():
        for X, Y in tqdm(dataloader, total=len(dataloader)):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)
            prob_outputs = logits.softmax(dim=1)

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

    # 计算并保存每个类别的平均 PLPD 值
    avg_class_plpd_values = {}
    for class_id in range(100):
        if class_plpd_values[class_id]:
            avg_class_plpd_values[class_id] = sum(class_plpd_values[class_id]) / len(class_plpd_values[class_id])
        else:
            avg_class_plpd_values[class_id] = 0

    metrics_file_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/avg_len_16_plpd_values_cifar100.pkl'
    with open(metrics_file_path, 'wb') as f:
        pickle.dump(avg_class_plpd_values, f)

    for class_id, avg_plpd in avg_class_plpd_values.items():
        print(f"类别 {class_id} 的平均 PLPD 值: {avg_plpd}")
def main():
    set_random_seed(42)

    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    patch_len = 16

    calculate_and_save_metrics(patch_len, device)

if __name__ == '__main__':
    main()