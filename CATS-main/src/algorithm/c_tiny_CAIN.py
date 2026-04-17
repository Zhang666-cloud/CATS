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

def load_checkpoint(model, model_path, device):
    try:
        # 先用torch.load尝试
        checkpoint = torch.load(model_path, map_location=device)
        print("Loaded checkpoint with torch.load")
        # 尝试自动识别模型权重字典
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise ValueError("Checkpoint is not a dict")
    except Exception as e:
        print(f"torch.load failed: {e}")
        # 尝试pickle加载
        import pickle
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
            print("Loaded checkpoint with pickle.load")
            # 你需要打印checkpoint结构，自行调整载入方式
            print(f"Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise ValueError("Checkpoint format not recognized for loading state dict")

def calculate_and_save_metrics_tiny_imagenet(patch_len, device, dataset_path, model_path):
    args = args_parser()
    model = create_model(args)
    model.to(device)

    load_checkpoint(model, model_path, device)

    model.change_bn(mode='grad')
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])

    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_plpd_values = defaultdict(list)

    with torch.no_grad():
        for X, Y in tqdm(dataloader, total=len(dataloader)):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)
            prob_outputs = logits.softmax(dim=1)

            x_prime = X.clone().detach()
            h, w = x_prime.shape[-2], x_prime.shape[-1]
            new_h = (h // patch_len) * patch_len
            new_w = (w // patch_len) * patch_len

            resize_t = transforms.Resize((new_h, new_w))
            resize_o = transforms.Resize((h, w))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
            x_prime = resize_o(x_prime)

            outputs_prime = model(x_prime)
            prob_outputs_prime = outputs_prime.softmax(1)
            cls1 = prob_outputs.argmax(dim=1)

            plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(
                prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
            plpd = plpd.reshape(-1)

            for j in range(len(Y)):
                class_id = Y[j].item()
                class_plpd_values[class_id].append(plpd[j].item())

    num_classes = 200
    avg_class_plpd_values = {}
    for class_id in range(num_classes):
        if class_plpd_values[class_id]:
            avg_class_plpd_values[class_id] = sum(class_plpd_values[class_id]) / len(class_plpd_values[class_id])
        else:
            avg_class_plpd_values[class_id] = 0

    save_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/blur_avg_len_8_plpd_values_tiny_imagenet_train.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(avg_class_plpd_values, f)

    for class_id, avg_plpd in avg_class_plpd_values.items():
        print(f"Class {class_id} average PLPD value: {avg_plpd}")

def main():
    set_random_seed(42)
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    patch_len = 8
    dataset_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/data/tiny-imagenet-200/train'
    model_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/weights/tiny_imagenet/feat/blur_pretrain_fedavg_resnet18_pseed_0_seed_0.pkl'

    calculate_and_save_metrics_tiny_imagenet(patch_len, device, dataset_path, model_path)

if __name__ == '__main__':
    main()