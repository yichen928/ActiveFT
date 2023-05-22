# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits


def extract_features(args):
    # utils.init_distributed_mode(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit', args.arch, num_classes=0)
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    if 'dino' in args.pretrained_weights or args.pretrained_weights == "":
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    else:
        ckpt_dict = torch.load(args.pretrained_weights)
        if "model" in ckpt_dict:
            ckpt_dict = ckpt_dict["model"]
        if "state_dict" in ckpt_dict:
            ckpt_dict = ckpt_dict["state_dict"]

        new_ckpt_dict = {}
        for key in ckpt_dict:
            if "head" not in key:
                if "backbone" in key:
                    new_key = key[9:]
                else:
                    new_key = key
                new_ckpt_dict[new_key] = ckpt_dict[key]

        model.load_state_dict(new_ckpt_dict, strict=True)
    print(f"Model {args.arch} built.")

    # ============ preparing data ... ============
    data_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # data_transform = pth_transforms.Compose([
    #     pth_transforms.ToTensor(),
    #     pth_transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.200, 0.201)),
    # ])

    if args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(root="data", train = args.split=="train", download=True, transform=data_transform)
    elif args.dataset == "CIFAR100":
        dataset = datasets.CIFAR100(root="data", train = args.split=="train", download=True, transform=data_transform)
    elif args.dataset == "ImageNet":
        root = os.path.join("data/ImageNet", 'train' if args.split=="train" else 'val')
        dataset = datasets.ImageFolder(root, transform=data_transform)
    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Data loaded with {len(dataset)} imgs.")

    features, labels = validate_network(data_loader, model, args.n_last_blocks, args.avgpool_patchtokens)
    labels = labels.reshape(-1, 1)

    outputs = np.concatenate([features, labels], axis=1)

    np.save(os.path.join(args.output_dir, "%s%s%s.npy"%(args.dataset, args.extra_name, args.split)), outputs)


@torch.no_grad()
def validate_network(data_loader, model, n, avgpool):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    features = None
    targets = None
    for inp, target in metric_logger.log_every(data_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)

        output = output.cpu().numpy()
        target = target.numpy()

        if features is None:
            features = output
            targets = target
        else:
            features = np.concatenate([features, output], axis=0)
            targets = np.concatenate([targets, target], axis=0)

    return features, targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token. 
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help='Dataset to extract features')
    parser.add_argument('--output_dir', default="features", help='Path to save extracted features')

    parser.add_argument('--split', default="train", type=str, help="Data split to extract features")
    parser.add_argument('--extra_name', default="_", type=str, help="extra filename")

    args = parser.parse_args()
    extract_features(args)
