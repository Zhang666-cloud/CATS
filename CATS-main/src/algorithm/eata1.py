import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer

class EataServer(TTABaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: EataClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: EataClient(cid, datasets, args) for cid, datasets in test_datasets.items()}
        # load a pre-trained model (loading in main.py)
        self.model = create_model(args)
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

                # print('one')
class EataClient(BaseClient):
    def local_eval(self, model, args, dataset='test'):
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)
        e_margin =0.1*math.log(200)
        d_margin=0.05
        current_model_probs=None
        total_examples, total_loss, total_metric = 0, 0, 0

        for *X, Y in self.dataloaders[dataset]:
            # Get a batch of data
            # for i in range(6):
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            logits = model(*X)
            entropys = softmax_entropy(logits)
            filter_ids_1 = torch.where(entropys < e_margin)

            # # 获取样本数量
            # num_samples = entropys.size(0)
            #
            # # 计算前 40% 的数量
            # top_40_percent_count = int(num_samples * 0.2)
            #
            # # 对熵进行排序，获取排序后的索引
            # sorted_entropy_indices = torch.argsort(entropys)
            #
            # # 获取前 40% 的索引
            # filter_ids_1 = sorted_entropy_indices[:top_40_percent_count]
            ids1 = filter_ids_1

            ids2 = torch.where(ids1[0] > -0.1)
            entropys = entropys[filter_ids_1]
            if current_model_probs is not None:
                cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0),
                                                          logits[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(current_model_probs, logits[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(current_model_probs, logits[filter_ids_1].softmax(1))
            current_model_probs = updated_probs
            coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
            # """
            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
            loss = entropys.mean(0)
            filtered_X = [x[ids1][ids2] for x in X]  # 对列表中的每个张量应用索引
            if any(f.size(0) != 0 for f in filtered_X):  # 检查是否有非空张量
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            with torch.no_grad():
                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)
                num_examples = len(X[0])

                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
        return avg_loss, avg_metric, total_examples
def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x