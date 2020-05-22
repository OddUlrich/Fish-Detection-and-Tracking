#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pwd + "..")

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tools.utils import load_classes
from tools.parse_config import parse_data_config, parse_model_config
from tools.datasets import ListDataset
from models import Darknet


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads used during batch generation")
parser.add_argument("--is_cuda", type=bool, default=True, help="whether to use cuda")
parser.add_argument("--class_path", type=str, default="data/fish.names", help="path to class label file")
parser.add_argument("--data_config_path", type=str, default="data/fish.data", help="path to data config file")
parser.add_argument("--model_config_path", type=str, default="config/fish.cfg", help="path to model config file")
parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to pre-trained weight file")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")

args = parser.parse_args()
#print(args)

# Load data class and data configuration.
classes = load_classes(args.class_path)
data_cfg = parse_data_config(args.data_config_path)
train_path = data_cfg["train"]

# Load hyperparameters.
hyperparams = parse_model_config(args.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])


###################################################################################3

# Create and initial model.
model = Darknet(args.model_config_path)
model.load_weights(args.weights_path)

# GPU flag.
cuda = torch.cuda.is_available() and args.is_cuda
if cuda:
    model = model.cuda()
model.train()

# Get dataloader and setop tensor and optimizer.
dataloader = DataLoader(ListDataset(train_path), 
                         batch_size=args.batch_size, 
                         shuffle=False, 
                         num_workers=args.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

os.makedirs("checkpoints", exist_ok=True)
# Training epochs.
for epoch in range(args.epochs):
    for idx_batch, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
            
        optim.zero_grad()
        loss = model(imgs, targets)
        loss.backward()
        optim.step()
        
        print("[Epoch %d/%d, Batch %d/%d] Loss: x=%f, y=%f, w=%f, h=%f, conf=%f, cls=%f, total=%f, recall: %.5f, precision: %.5f"
            % (epoch, args.epochs, idx_batch, len(dataloader), 
                  model.losses["x"], model.losses["y"], model.losses["w"], model.losses["h"],
                  model.losses["conf"], model.losses["cls"],
                  loss.item(), model.losses["recall"], model.losses["precision"]
              )
            )
        
        model.seen += imgs.size(0)

    if epoch % args.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (args.checkpoint_dir, epoch))






