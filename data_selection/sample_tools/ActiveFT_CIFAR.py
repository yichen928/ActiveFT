import json
import os
import numpy as np
import random
import argparse
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *

torch.autograd.set_detect_anomaly(True)
eps = 1e-10
infty = 1e10


class SampleModel(nn.Module):
    def __init__(self, features, sample_num, temperature, init, distance, balance=1.0):
        super(SampleModel, self).__init__()
        self.features = features
        self.total_num = features.shape[0]
        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance

        self.init = init
        self.distance = distance

        centroids = self.init_centroids()
        self.centroids = nn.Parameter(centroids).cuda()
        print(self.centroids.shape)

    def init_centroids(self):
        if self.init == "random":
            sample_ids = list(range(self.total_num))
            sample_ids = random.sample(sample_ids, self.sample_num)
        elif self.init == "fps":
            dist_func = functools.partial(get_distance, type=self.distance)
            sample_ids = farthest_distance_sample(self.features, self.sample_num, dist_func)

        centroids = self.features[sample_ids].clone()
        return centroids

    def get_loss(self):
        centroids = F.normalize(self.centroids, dim=1)
        prod = torch.matmul(self.features, centroids.transpose(1, 0))  # (n, k)
        prod = prod / self.temperature
        prod_exp = torch.exp(prod)
        prod_exp_pos, pos_k = torch.max(prod_exp, dim=1)  # (n, )

        cent_prod = torch.matmul(centroids.detach(), centroids.transpose(1, 0))  # (k, k)
        cent_prod = cent_prod / self.temperature
        cent_prod_exp = torch.exp(cent_prod)
        cent_prob_exp_sum = torch.sum(cent_prod_exp, dim=0)  # (k, )

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)

        return J


def optimize_dist(features, sample_num, args):
    #  features: (n, c)
    sample_model = SampleModel(features, sample_num, args.temperature, args.init, args.distance, args.balance)
    sample_model = sample_model.cuda()

    optimizer = optim.Adam(sample_model.parameters(), lr=args.lr)
    if args.scheduler != "none":
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iter, eta_min=1e-6)
        else:
            raise NotImplementedError

    for i in range(args.max_iter):
        loss = sample_model.get_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scheduler != "none":
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print("Iter: %d, lr: %.6f, loss: %f" % (i, lr, loss.item()))

    centroids = sample_model.centroids.detach()
    centroids = F.normalize(centroids, dim=1)
    dist = torch.matmul(centroids, features.transpose(1, 0))  # (k, n)
    _, sample_ids = torch.max(dist, dim=1)
    sample_ids = sample_ids.cpu().numpy().tolist()
    print(len(sample_ids), len(set(sample_ids)))
    sample_ids = set()
    _, ids_sort = torch.sort(dist, dim=1, descending=True)
    for i in range(ids_sort.shape[0]):
        for j in range(ids_sort.shape[1]):
            if ids_sort[i, j].item() not in sample_ids:
                sample_ids.add(ids_sort[i, j].item())
                break
    print(len(sample_ids))
    sample_ids = list(sample_ids)
    return sample_ids


def main(args):
    input = np.load(args.feature_path)
    features, _ = input[:, :-1], input[:, -1]
    features = torch.Tensor(features).cuda()

    total_num = features.shape[0]
    sample_num = int(total_num * args.percent * 0.01)

    if args.filename is None:
        name = args.feature_path.split("/")[-1]
        name = name[:-4]
        if args.balance != 1:
            args.filename = name + "_ActiveFT_%s_temp_%.2f_lr_%f_scheduler_%s_br_%.2f_iter_%d_sampleNum_%d.json" % (
                args.distance, args.temperature, args.lr, args.scheduler, args.balance, args.max_iter, sample_num)
        else:
            args.filename = name + "_ActiveFT_%s_temp_%.2f_lr_%f_scheduler_%s_iter_%d_sampleNum_%d.json" % (
                args.distance, args.temperature, args.lr, args.scheduler, args.max_iter, sample_num)
    output_path = os.path.join(args.output_dir, args.filename)

    features = F.normalize(features, dim=1)
    sample_ids = optimize_dist(features, sample_num, args)
    sample_ids.sort()
    print(output_path)
    with open(output_path, "w") as file:
        json.dump(sample_ids, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('--feature_path', default='features/CIFAR10_train.npy', type=str,
                        help='path of saved features')
    parser.add_argument('--output_dir', default='features', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for softmax')
    parser.add_argument('--threshold', default=0.0001, type=float, help='convergence threshold')
    parser.add_argument('--max_iter', default=300, type=int, help='max iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--percent', default=2, type=float, help='sample percent')
    parser.add_argument('--init', default='random', type=str, choices=['random', 'fps'])
    parser.add_argument('--distance', default='euclidean', type=str, help='euclidean or cosine')
    parser.add_argument('--scheduler', default='none', type=str, help='scheduler')
    parser.add_argument('--balance', default=1.0, type=float, help='balance ratio')
    args = parser.parse_args()
    main(args)