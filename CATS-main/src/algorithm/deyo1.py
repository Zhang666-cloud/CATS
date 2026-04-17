import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from copy import deepcopy
from einops import rearrange
from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer

class DeyoServer(TTABaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: DeyoClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: DeyoClient(cid, datasets, args) for cid, datasets in test_datasets.items()}
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
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
class DeyoClient(BaseClient):
    def local_eval(self, model, args, dataset='test'):
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)
        plpd_threshold=0.3
        reweight_ent=1
        reweight_plpd=0.005
        margin =0.2*math.log(100)
        deyo_margin = 0.4 * math.log(100)
        patch_len=4
        total_examples, total_loss, total_metric = 0, 0, 0
        for *X, Y in self.dataloaders[dataset]:
            original_X = torch.cat([x.to(self.device) for x in X], dim=0)
            original_Y = Y.to(self.device)
            # for i in range(6):
            X = original_X.clone()
            Y = original_Y.clone()

            logits = model(X)
            optimizer.zero_grad()
            entropys = softmax_entropy(logits)
            filter_ids_1 = torch.where((entropys < deyo_margin))
            entropys = entropys[filter_ids_1]

            if isinstance(X, list):
                X = torch.stack([x for x in X], dim=0)

            x_prime = X[filter_ids_1]
            x_prime = x_prime.detach()

            resize_t = torchvision.transforms.Resize(((X.shape[-1]//patch_len)*patch_len,(X.shape[-1]//patch_len)*patch_len))
            resize_o = torchvision.transforms.Resize((X.shape[-1],X.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
            x_prime = resize_o(x_prime)

            with torch.no_grad():
                outputs_prime = model(x_prime)
            prob_outputs = logits[filter_ids_1].softmax(1)
            prob_outputs_prime = outputs_prime.softmax(1)


            cls1 = prob_outputs.argmax(dim=1)
            plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(prob_outputs_prime,
                                                                                               dim=1,
                                                                                               index=cls1.reshape(-1,
                                                                                                                  1))
            plpd = plpd.reshape(-1)
            filter_ids_2 = torch.where(plpd > plpd_threshold)

            entropys = entropys[filter_ids_2]
            final_backward = len(entropys)

            plpd = plpd[filter_ids_2]
            coeff = (reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                     reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                     )
            entropys = entropys.mul(coeff)
            loss = entropys.mean(0)
            if final_backward != 0:
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
