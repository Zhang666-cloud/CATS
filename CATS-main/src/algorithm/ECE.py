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


class ECETestServer(BaseServer):

    def __init__(self, train_datasets, test_datasets, args):

        BaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: ECETestClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ECETestClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

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
        eces = []  # 添加ECE列表

        if mode == 'valid':
            clients = self.train_clients
        else:
            clients = self.test_clients

        for cid, client in tqdm(clients.items()):
            loss, metric, ece, num_data = client.local_eval(self.model, self.adaptation_rates, args,
                                                            'test')  # 修改为接收4个返回值
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)
            eces.append(ece)  # 添加ECE

            # reset the model (the adaptation rate is not update, do not need to reset)
            self.model.load_state_dict(global_state, strict=False)

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        agg_ece = sum([weight * ece for weight, ece in zip(weights, eces)]) / sum(weights)  # 计算加权平均ECE

        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f \t ECE: %.4f' % (agg_loss, agg_metric, agg_ece))  # 添加ECE输出

        log_dict = {
            mode + '_losses': losses,
            mode + '_metrics': metrics,
            mode + '_eces': eces,  # 添加ECE记录
            mode + '_wavg_loss': agg_loss,
            mode + '_wavg_metric': agg_metric,
            mode + '_wavg_ece': agg_ece,  # 添加平均ECE记录
        }
        self.history.append(log_dict)


class ECETestClient(BaseClient):
    def check_file_contents(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(data.keys())  # 打印所有键以检查其内容

    def adapt_one_step(self, model, adapt_lrs, *X, Y, unspv_loss_func, args, all_plpd, outputs_prime_fil, y2):
        """
        实现EATA-C校准方法，在测试时进行适应。仅更新完整模型中子网络未跳过的参数。
        """
        model.to(args.device)
        model.eval()  # 切换完整模型至评估模式
        margin = 0.1 * math.log(10)
        # model.save_snapshot()  # 在开始训练前保存初始快照

        # 获取完整网络的预测
        logits_full = model(*X)
        entory_margin = 0.8 * math.log(200)
        # 创建子网络并获取其预测
        subnetwork = model.get_subnetwork().to(args.device)

        subnetwork.eval()
        with torch.no_grad():
            logits_sub = subnetwork(*X)

        # 计算概率分布
        probs_full = torch.softmax(logits_full, dim=1)
        probs_sub = torch.softmax(logits_sub, dim=1)

        # 使用标签平滑的融合预测
        p = 0.1  # 标签平滑参数
        probs_fuse = (probs_full + (1 - p) * probs_sub) / (2 - p)

        # 计算KL散度损失
        consistency_loss = nn.functional.kl_div(
            nn.functional.log_softmax(logits_sub, dim=1),
            probs_fuse,
            reduction='batchmean'
        )

        # 计算最小最大熵正则化
        C = torch.where(
            torch.argmax(probs_full, dim=1) == torch.argmax(probs_sub, dim=1),
            torch.tensor(1.0).to(probs_full.device),  # 标签一致
            torch.tensor(-1.0).to(probs_full.device)  # 标签不一致
        )

        entropy = unspv_loss_func(logits_sub, None)
        filter_ids_1 = torch.where((entropy < entory_margin))
        entropy=entropy[filter_ids_1]
        coeff = (1 * (1 / (torch.exp(((entropy.clone().detach()) - margin)))) +
                 0.0001 * (1 / (torch.exp(-1. * all_plpd.clone().detach())))
                 )

        minmax_entropy_loss = (C * entropy)
        minmax_entropy_loss=minmax_entropy_loss.mul(coeff)
        minmax_entropy_loss=minmax_entropy_loss.mean(dim=0)

        # 计算总损失
        alpha = 0.1  # 熵正则化的权重
        loss = consistency_loss + alpha * minmax_entropy_loss
        # 反向传播损失
        loss.backward()

        model.set_running_stat_grads()

        # 用于计算子网络中未跳过参数的梯度
        unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]
        with torch.no_grad():
            # for p_full, p_sub in zip(
            #         model.trainable_parameters(),
            #         subnetwork.trainable_parameters()
            # ):
                # if p_sub.grad is not None:
                #     # 对于计算的梯度保持更新
                #     unspv_grad.append(p_sub.grad.clone())
                # else:
                #     # 非常量的梯度为0
                #     unspv_grad.append(torch.zeros_like(p_full))
            # 更新完整模型中的参数
            for i, (p_full, g_sub) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
                # 仅在子网络未跳过的部分进行更新
                if g_sub is not None:
                    p_full -= adapt_lrs[i] * g_sub
                else:
                    print(1)

        # 重置梯度
        model.zero_grad()

        # 修正BatchNorm层运行时变量
        model.clip_bn_running_vars()

        return unspv_grad

    def calculate_ece(self, logits, labels, n_bins=15):
        """
        计算Expected Calibration Error (ECE)
        """
        # 获取预测概率和预测类别
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        # 分桶计算
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 确定当前桶中的样本
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                # 计算桶内的准确率和平均置信度
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                # 累加ECE
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()

    def local_eval(self, model, adapt_lrs, args, dataset='test'):
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        current_lrs = adapt_lrs.clone()
        patch_len = 16
        total_examples, total_loss, total_metric, total_ece = 0, 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]
        state = deepcopy(model.state_dict())
        class_plpd_values = load_class_plpd(
            '/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/avg_len_16_plpd_values_cifar100.pkl')
        # self.check_file_contents('/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/avg_len_4_plpd_values_cifar100.pkl')
        all_logits = []
        all_labels = []
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
                state_start = wavg_state(state, state_now, 1 / (i + 1))
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

                # 保存打乱后的预测类别y2
                y2 = outputs_prime.argmax(dim=1)

                # 计算实际 PLPD
                plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(
                    prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
                plpd = plpd.reshape(-1)

                topk_probs, topk_indices = prob_outputs.topk(3, dim=1)

                # 计算加权 PLPD 阈值
                threshold_plpd = []
                for i in range(prob_outputs.size(0)):
                    weight_sum = topk_probs[i].sum()
                    weighted_plpd = 0
                    for j in range(3):
                        class_id = topk_indices[i, j].item()
                        weighted_plpd += (topk_probs[i, j] / weight_sum) * class_plpd_values[class_id]
                    threshold_plpd.append(weighted_plpd)

                threshold_plpd = torch.tensor(threshold_plpd, device=X.device)
                threshold_plpd *= 0.001
                # 筛选基于加权 PLPD 阈值
                filtered_indices = plpd > threshold_plpd
                plpd = plpd[filtered_indices]
                filtered_X = original_X[filtered_indices]
                filtered_Y = original_Y[filtered_indices]

                # 保存筛选后的y1和y2
                y1_filtered = cls1[filtered_indices]
                y2_filtered = y2[filtered_indices]
                outputs_prime_fil=outputs_prime[filtered_indices]
            # 调用适应步骤，传入y1和y2
            self.adapt_one_step(model, current_lrs, filtered_X, Y=filtered_Y, unspv_loss_func=unspv_loss_func,
                                args=args, all_plpd=plpd, outputs_prime_fil=outputs_prime_fil, y2=y2_filtered)

            if args.test == 'online_avg':
                state_now = model.state_dict()
                state_new = wavg_state(acc_state, state_now, i / (i + 1))
                model.load_state_dict(state_new)

            # 2. supervised evaluation
            model.eval()
            with torch.no_grad():
                logits = model(X)
                spv_loss = spv_loss_func(logits, Y)
                # 计算当前batch的ECE
                batch_ece = self.calculate_ece(logits, Y)

                # 收集所有logits和labels用于整体ECE计算
                all_logits.append(logits)
                all_labels.append(Y)
                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples
                total_ece += batch_ece * num_examples  # 累加ECE

        if all_logits:
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            overall_ece = self.calculate_ece(all_logits, all_labels)
        else:
            overall_ece = 0

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
        avg_ece = total_ece / total_examples  # 计算平均ECE

        # 返回平均损失、准确率、ECE和数据量
        return avg_loss, avg_metric, avg_ece, num_data


def wavg_state(state1, state2, lamda):
    state = deepcopy(state1)
    for k in state1.keys():
        state[k] = lamda * state1[k] + (1 - lamda) * state2[k]

    return state


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # temprature = 1.1 #0.9 #1.2
    # x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)