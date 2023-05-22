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
import math
import utils

torch.autograd.set_detect_anomaly(True)
eps = 1e-10
infty = 1e10


class SampleModel(nn.Module):
    def __init__(self, features, sample_num, temperature, init, distance, balance, slice, batch_size):
        super(SampleModel, self).__init__()
        self.features = features
        self.total_num = features.shape[0]
        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance
        self.slice = slice
        if slice is None:
            self.slice = self.total_num
        self.batch_size = batch_size

        self.init = init
        self.distance = distance

        centroids = self.init_centroids()
        self.centroids = nn.Parameter(centroids).cuda()

    def init_centroids(self):
        if self.init == "random":
            sample_ids = list(range(self.total_num))
            sample_ids = random.sample(sample_ids, self.sample_num)
        elif self.init == "fps":
            dist_func = functools.partial(utils.get_distance, type=self.distance)
            sample_ids = utils.farthest_distance_sample(self.features, self.sample_num, dist_func)

        centroids = self.features[sample_ids].clone()
        return centroids

    def get_loss(self):
        centroids = F.normalize(self.centroids, dim=1)
        sample_ids = list(range(self.total_num))
        sample_ids = random.sample(sample_ids, self.batch_size)
        features = self.features[sample_ids]
        sample_slice_num = math.ceil(1.0 * self.sample_num / self.slice)
        batch_slice_num = math.ceil(1.0 * self.batch_size / self.slice)

        prod_exp_pos = []
        pos_k = []
        for sid in range(batch_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            prod = torch.matmul(features[start: end], centroids.transpose(1, 0))  # (slice_num, k)
            prod = prod / self.temperature
            prod_exp = torch.exp(prod)
            prod_exp_pos_cur, pos_k_cur = torch.max(prod_exp, dim=1)  # (slice_num, )
            prod_exp_pos.append(prod_exp_pos_cur)
            pos_k.append(pos_k_cur)
        pos_k = torch.cat(pos_k, dim=0)
        prod_exp_pos = torch.cat(prod_exp_pos, dim=0)

        cent_prob_exp_sum = []
        for sid in range(sample_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            cent_prod = torch.matmul(centroids.detach(), centroids[start:end].transpose(1, 0))  # (k, slice_num)
            cent_prod = cent_prod / self.temperature
            cent_prod_exp = torch.exp(cent_prod)
            cent_prob_exp_sum_cur = torch.sum(cent_prod_exp, dim=0)  # (slice_num, )
            cent_prob_exp_sum.append(cent_prob_exp_sum_cur)
        cent_prob_exp_sum = torch.cat(cent_prob_exp_sum, dim=0)

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)

        return J


def optimize_dist(features, sample_num, args):
    #  features: (n, c)
    sample_model = SampleModel(features, sample_num, args.temperature, args.init, args.distance, args.balance, args.slice, args.batch_size)
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
    slice = 100
    sample_slice_num = math.ceil(centroids.shape[0] / slice)
    sample_ids = set()
    # _, ids_sort = torch.sort(dist, dim=1, descending=True)
    for sid in range(sample_slice_num):
        start = sid * slice
        end = min((sid + 1) * slice, centroids.shape[0])
        dist = torch.matmul(centroids[start:end], features.transpose(1, 0))  # (slice_num, n)
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
    parser.add_argument('--feature_path', default='features/ImageNet_train.npy', type=str,
                        help='path of saved features')
    parser.add_argument('--output_dir', default='features', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for softmax')
    parser.add_argument('--threshold', default=0.0001, type=float, help='convergence threshold')
    parser.add_argument('--max_iter', default=100, type=int, help='max iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--percent', default=1, type=float, help='sample percent')
    parser.add_argument('--init', default='random', type=str, choices=['random', 'fps'])
    parser.add_argument('--distance', default='euclidean', type=str, help='euclidean or cosine')
    parser.add_argument('--scheduler', default='none', type=str, help='scheduler')
    parser.add_argument('--balance', default=1.0, type=float, help='balance ratio')
    parser.add_argument('--batch_size', default=100000, type=int, help='batch size for SGD')
    parser.add_argument('--slice', default=None, type=int, help='size of slice to save memory')
    args = parser.parse_args()
    main(args)