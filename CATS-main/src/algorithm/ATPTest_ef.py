import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer
from model.MyBatchNorm2d import MyBatchNorm2d
from utils import pickle_load
import pickle
from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer
def load_class_plpd(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class ATPTestef_Server(BaseServer):


    def __init__(self, train_datasets, test_datasets, args):

        BaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: ATPTestef_Client(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPTestef_Client(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        self.model.change_bn(mode='grad')  # replace the nn.BatchNorm2d to our BatchNorm,
        # which has identical behavior, but support taking gradient
        self.model.eval()

        self.adaptation_rates = self.load_adapt_lrs(args)


    def load_adapt_lrs(self, args):
        path = args.load_adapt_path
        idx = args.load_adapt_idx
        rnd = args.load_adapt_round

        if path == 'manual':
            rate = torch.zeros(102).to(args.device)
            lr = args.lm_lr
            m = args.batchadapt_bn_momentum

            # stats_idx = [] # 1, 2, 6, 7, ..., 96, 97
            # for i in range(20):
            #     stats_idx.append(i * 5 + 1)
            #     stats_idx.append(i * 5 + 2)

            if args.layers_to_adapt == 'none':
                pass

            elif args.layers_to_adapt == 'const':
                rate = torch.ones(102).to(args.device) * lr

            elif args.layers_to_adapt == 'first_conv_bn':
                params_idxs = [0, 3, 4]
                stats_idxs = [1, 2]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'block1':
                params_idxs = [5, 8, 9, 10, 13, 14, 15, 18, 19, 20, 23, 24]
                stats_idxs = [6, 7, 11, 12, 16, 17, 21, 22]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'block2':
                params_idxs = [25, 28, 29, 30, 33, 34, 35, 38, 39, 40, 43, 44, 45, 48, 49]
                stats_idxs = [26, 27, 31, 32, 36, 37, 41, 42, 46, 47]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'block3':
                params_idxs = [50, 53, 54, 55, 58, 59, 60, 63, 64, 65, 68, 69, 70, 73, 74]
                stats_idxs = [51, 52, 56, 57, 61, 62, 66, 67, 71, 72]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'block4':
                params_idxs = [75, 78, 79, 80, 83, 84, 85, 88, 89, 90, 93, 94, 95, 98, 99]
                stats_idxs = [76, 77, 81, 82, 86, 87, 91, 92, 96, 97]

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'last_layer':
                params_idxs = [100, 101]
                for idx in params_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'all_bn':
                params_idxs = []  # 3, 4, 8, 9, ..., 98, 99
                for i in range(20):
                    params_idxs.append(5 * i + 3)
                    params_idxs.append(5 * i + 4)

                stats_idxs = []  # 1, 2, 6, 7, ..., 96, 97
                for i in range(20):
                    stats_idxs.append(i * 5 + 1)
                    stats_idxs.append(i * 5 + 2)

                for idx in params_idxs:
                    rate[idx] = lr
                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'all_bn_stats':

                stats_idxs = []  # 1, 2, 6, 7, ..., 96, 97
                for i in range(20):
                    stats_idxs.append(i * 5 + 1)
                    stats_idxs.append(i * 5 + 2)

                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'all_bn_running_mean':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5 + 1)

                for idx in stats_idxs:
                    rate[idx] = m

            elif args.layers_to_adapt == 'all_bn_running_var':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5 + 2)

                for idx in stats_idxs:
                    rate[idx] = m


            elif args.layers_to_adapt == 'all_bn_weight':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5 + 3)

                for idx in stats_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'all_bn_bias':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5 + 4)

                for idx in stats_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'all_conv':

                stats_idxs = []
                for i in range(20):
                    stats_idxs.append(i * 5)

                for idx in stats_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'last_weight':
                params_idxs = [100, ]
                for idx in params_idxs:
                    rate[idx] = lr

            elif args.layers_to_adapt == 'last_bias':
                params_idxs = [101, ]
                for idx in params_idxs:
                    rate[idx] = lr



            print(rate)


        elif path == 'zero':
            rate = torch.zeros(102).to(args.device)

        else:
            data = pickle_load(path, True)[idx]
            rate = data['history']['adapt_lrs'][rnd]
            rate = torch.Tensor(rate).to(args.device)

        return rate


    def run(self, args):
        # for rnd in range(1, 6):
            # No Training, Direct Evaluation
            # self.adapt_and_eval(args, 'valid')
            self.adapt_and_eval(args, 'test')

    def adapt_and_eval(self, args, mode='test'):
        # current global model
        global_state = deepcopy(self.model.updated_state_dict())

        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        if mode == 'valid':
            clients = self.train_clients
        else:
            clients = self.test_clients

        for cid, client in tqdm(clients.items()):
            loss, metric, num_data = client.local_eval(self.model, self.adaptation_rates, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            # reset the model (the adaptation rate is not update, do not need to reset)
            self.model.load_state_dict(global_state, strict=False)

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            mode + '_losses': losses,
            mode + '_metrics': metrics,
            mode + '_wavg_loss': agg_loss,
            mode + '_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

class ATPTestef_Client(BaseClient):
    def check_file_contents(self,file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(data.keys())  # 打印所有键以检查其内容
    def adapt_one_step(self, model, adapt_lrs, *X, Y, unspv_loss_func, args,all_plpd):

        model.eval()

        logits = model(*X)
        entory_margin = 0.8 * math.log(200)
        loss = unspv_loss_func(logits, Y)
        filter_ids_1 = torch.where((loss < entory_margin))
        loss = loss[filter_ids_1]
        # if len(all_plpd) != len(loss):
        #     # 如果长度不匹配，使用相同的过滤条件处理 all_plpd
        #     all_plpd = all_plpd[filter_ids_1]
        # margin = 0.1* math.log(10)
        # coeff = (1 * (1 / (torch.exp(((loss.clone().detach()) - margin)))) +
        #          0.0001 * (1 / (torch.exp(-1. * all_plpd.clone().detach())))
        #          )
        # loss = loss.mul(coeff)
        loss = loss.mean(dim=0)
        loss.backward()

        model.set_running_stat_grads()

        unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
                p -= adapt_lrs[i] * g

        model.zero_grad()

        model.clip_bn_running_vars()  # some BN running vars may be smaller than 0, which cause NaN problem.

        return unspv_grad

    def local_eval(self, model, adapt_lrs, args, dataset='test'):
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        current_lrs = adapt_lrs.clone()
        patch_len = 8
        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]
        state = deepcopy(model.state_dict())
        class_plpd_values = load_class_plpd('/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/weather_avg_len_8_plpd_values_tiny_imagenet_train.pkl')
        # self.check_file_contents('/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/avg_len_4_plpd_values_cifar100.pkl')
        for i, (*X, Y) in enumerate(dataloader):

            if args.test == 'batch':
                # Use the same global client for all batches
                model.load_state_dict(state)

            elif args.test == 'large_batch':
                # Use the same global client for all batches
                model.load_state_dict(state)
                current_lrs = adapt_lrs

            elif args.test == 'online_raw':
                # Directly pass the current model for next batch
                pass

            elif args.test == 'online_small':
                current_lrs = adapt_lrs / 10

            elif args.test == 'online_exp':
                current_lrs = adapt_lrs * (0.5 ** i) * 0.6667

            elif args.test == 'online':
                state_now = model.state_dict()
                state_start = wavg_state(state, state_now, 0.5)
                model.load_state_dict(state_start)

                current_lrs = adapt_lrs * 0.5

            elif args.test == 'online_ha':
                state_now = model.state_dict()
                state_start = wavg_state(state, state_now, 1 / (i+1))
                model.load_state_dict(state_start)

                current_lrs = adapt_lrs / (i + 1)

            elif args.test == 'online_avg':

                if i == 0:
                    acc_state = deepcopy(state)  # the average of all previous state

                else:  # i > 0
                    acc_state = deepcopy(model.state_dict())  # the average of all previous state
                    model.load_state_dict(state)
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)
            model.to(args.device)
            # Get a batch of data
            original_X = torch.cat([x.to(self.device) for x in X], dim=0)
            original_Y = Y.to(self.device)
            plpd_threshold = -100
            # for i in range(6):
            X = original_X.clone()
            Y = original_Y.clone()
            with torch.no_grad():
                logits = model(X)
                x_prime = original_X.clone().detach()
                resize_t = torchvision.transforms.Resize(
                    ((x_prime.shape[-1] // patch_len) * patch_len, (x_prime.shape[-1] // patch_len) * patch_len))
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

                # 计算实际 PLPD
                plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(
                    prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
                plpd = plpd.reshape(-1)

                topk_probs, topk_indices = prob_outputs.topk(3, dim=1)

                filtered_indices = plpd > plpd_threshold
                plpd=plpd[filtered_indices]
                filtered_X = original_X[filtered_indices]
                filtered_Y = original_Y[filtered_indices]


            # 调用适应步骤
            self.adapt_one_step(model, current_lrs, filtered_X, Y=filtered_Y, unspv_loss_func=unspv_loss_func,
                                args=args,all_plpd=plpd)

            if args.test == 'online_avg':
                state_now = model.state_dict()
                state_new = wavg_state(acc_state, state_now, i / (i+1))
                model.load_state_dict(state_new)





            # 1. unsupervised adaptation



            # 2. supervised evaluation

            model.eval()
            with torch.no_grad():
                logits = model(X)
                spv_loss = spv_loss_func(logits, Y)

                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data


def wavg_state(state1, state2, lamda):
    state = deepcopy(state1)
    for k in state1.keys():
        state[k] = lamda * state1[k] + (1 - lamda) * state2[k]

    return state

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)