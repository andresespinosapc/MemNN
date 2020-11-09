#!/usr/bin/python
from __future__ import division
from Model import MLP
from config import config
from utils import *
import pdb
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

train_q, train_a, train_sim = load_from_file("./pkl/sim/train_pair.pkl")
dev_q, dev_a, dev_sim = load_from_file("./pkl/sim/dev_pair.pkl")

assert len(train_q) == len(train_a) == len(train_sim)
n_batches_per_epoch = int(len(train_q) / config.batch_size) + 1
n_batches = n_batches_per_epoch * config.epoch

train_q_iter = batch_sort_iter(train_q, config.batch_size, config.epoch, padding = True)
train_a_iter = batch_sort_iter(train_a, config.batch_size, config.epoch, padding = True, sort=False)
train_sim_iter = batch_sort_iter(train_sim, config.batch_size, config.epoch, padding = False)

# large dev evaluation will result in memory issue
N = 1000
dev_q_t = to_tensor(dev_q[:N], padding = True) 
dev_a_t = to_tensor(dev_a[:N], padding = True, sort=False)
dev_sim_t = torch.LongTensor(dev_sim[:N])

model = MLP(config)
optimizer = optim.SGD(model.parameters(), lr=config.lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
dev_q_t, dev_a_t, dev_sim_t = dev_q_t.to(device), dev_a_t.to(device), dev_sim_t.to(device)

#pdb.set_trace()

def train(): 
    cnt = 0
    pbar = tqdm(zip(train_q_iter, train_a_iter, train_sim_iter), total=n_batches)
    val_loss = -1
    for i_q, i_a, i_s in pbar:
        i_q, i_a, i_s = i_q.to(device), i_a.to(device), i_s.to(device)
        #pdb.set_trace()
        model.zero_grad()
        loss = model.forward(i_q, i_a, i_s)
        loss.backward()
        optimizer.step()
        train_loss = loss.data.sum().item()
        cnt += 1
        if cnt % config.valid_every == 0:
            loss = model.forward(dev_q_t, dev_a_t, dev_sim_t)
            val_loss = loss.data.sum().item()
        pbar.set_description('Train loss: {:.3f}. Val loss: {:.3f}'.format(train_loss, val_loss))
train()
model.save(config.pre_embed_file)
