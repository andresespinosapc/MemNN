#!/usr/bin/python
from __future__ import division
import os
import sys
from Model import KVMemoryReader
from config import config
import operator
from utils import *
import pdb
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_q, train_w, train_e_p, train_a = load_from_file("./pkl/reader/{}/train_pair.pkl".format(config.d_embed))
dev_q, dev_w, dev_e_p, dev_a = load_from_file("./pkl/reader/{}/dev_pair.pkl".format(config.d_embed))
test_q, test_w, test_e_p, test_a = load_from_file("./pkl/reader/{}/test_pair.pkl".format(config.d_embed))
#train_q, train_w, train_e_p, train_a = load_from_file("./pkl/toy/reader/train_pair.pkl")
#dev_q, dev_w, dev_e_p, dev_a = load_from_file("./pkl/toy/reader/dev_pair.pkl")

def modify(q, wiki, pos, ans, neg_sampling=False):
    tL = torch.LongTensor
    ret_q = []
    ret_key = []
    ret_value = []
    ret_cand = []
    ret_a = []
    possible_ans = np.unique(ans)
    pbar = tqdm(zip(q,wiki,pos,ans), total=len(q))
    for qu,w,p,a_ in pbar:
        # encoding the candidate
        can_dict = {}
        qu = qu.numpy()
        w = w.numpy()
        p = p.numpy()
        a_ = a_.numpy()

        # generate local candidate
        len_w = len(w)
        cand_ind = []
        for i in range(len_w):
            if w[i][1] not in can_dict:
                can_dict[w[i][1]] = len(can_dict)
            if w[i][p[i]] not in can_dict:
                can_dict[w[i][p[i]]] = len(can_dict)
        if a_[0] not in can_dict:
            continue
        else:
            sort_l = sorted(can_dict.items(), key=operator.itemgetter(1))
            cand_l = [x[0] for x in sort_l]
            if neg_sampling:
                possible_neg_ans = possible_ans[~np.isin(possible_ans, cand_l)]
                neg_ans = np.random.choice(possible_neg_ans, config.n_neg_samples)
                cand_l = cand_l + neg_ans.tolist()

            # split into key value format
            #pdb.set_trace()
            key_m, val_m = transKV(w,p)
            ret_q.append(tL(qu))
            ret_key.append(tL(key_m))
            ret_value.append(tL(val_m))
            ret_cand.append(tL(cand_l))
            ret_a.append(tL([can_dict[a_[0]]]))
    print(len(ret_q) / len(q))
    return ret_q, ret_key,ret_value,ret_cand, ret_a 

def transKV(sents, pos):
    unk = 2
    ret_k = []
    ret_v = []
    for sent, p in zip(sents,pos):
        k_ = sent[3:].tolist() + [unk]
        v_ = sent[1]
        #pdb.set_trace()
        ret_k.append(k_)
        ret_v.append(v_)
        #print(toSent(k_),toSent([v_]))

        k_ = [sent[1]] + sent[3:].tolist()
        v_ = sent[p] 
        ret_k.append(k_)
        ret_v.append(v_)
        #print(toSent(k_),toSent([v_]))
    return np.array(ret_k), np.array(ret_v)

def pad_batch(batch):
    questions, q_lengths, keys, key_num_lengths, key_word_lengths, values, cands, cand_lengths, answers = zip(*batch)
    questions = pad_sequence(questions, batch_first=True)
    q_lengths = torch.tensor(q_lengths)
    keys = pad_sequence(keys, batch_first=True)
    key_num_lengths = torch.tensor(key_num_lengths)
    key_word_lengths = torch.tensor(key_word_lengths)
    values = pad_sequence(values, batch_first=True)
    cands = pad_sequence(cands, batch_first=True)
    cand_lengths = torch.tensor(cand_lengths)
    answers = torch.tensor(answers)

    return questions, q_lengths, keys, key_num_lengths, key_word_lengths, values, cands, cand_lengths, answers

def get_data_lengths(questions, keys, candidates):
    # Returns: question_word_lengths, key_num_lengths, key_word_lengths, cand_lengths
    return [
        list(map(lambda q: q.shape[0], questions)),
        list(map(lambda k: k.shape[0], keys)),
        list(map(lambda k: k.shape[1], keys)),
        list(map(lambda c: c.shape[0], candidates))
    ]

def save_model(epoch):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(config.reader_model_dir, 'd{}_{}.torch'.format(config.d_embed, epoch)))

def train(epoch):
    for e_ in list(range(epoch)):
        if config.use_lr_decay and (e_ + 1) % 10 == 0:
            adjust_learning_rate(optimizer, e_)
        cnt = 0
        pbar = tqdm(train_dataloader)
        train_loss = 0
        for question, q_length, key, key_num_length, key_word_length, value, candidate, cand_length, answer in pbar:
            cnt += 1
            question, q_length, key = question.to(device), q_length.to(device), key.to(device)
            key_num_length, key_word_length, value = key_num_length.to(device), key_word_length.to(device), value.to(device)
            candidate, cand_length, answer = candidate.to(device), cand_length.to(device), answer.to(device)
            cand_score = model(question, q_length, key, key_num_length,
                          key_word_length, value, candidate, cand_length)
            loss = loss_function(cand_score, answer)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('Epoch: {}. Train loss: {:.3f}'.format(e_ + 1, train_loss / cnt))
        dev_accuracy = eval()
        test_accuracy = eval('test')
        print('Valid. acc.: {:.3f}. Test acc.: {:.3f}'.format(dev_accuracy, test_accuracy))
        if (e_ + 1) % config.save_every == 0:
            save_model(e_ + 1)

def adjust_learning_rate(optimizer, epoch):
    lr = config.lr / (2 ** (epoch // 10))
    print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def eval(dataset='dev'):
    correct = 0
    total = 0
    if dataset == 'dev':
        pbar = tqdm(dev_dataloader)
    elif dataset == 'test':
        pbar = tqdm(test_dataloader)
    for question, q_length, key, key_num_length, key_word_length, value, candidate, cand_length, answer in pbar:
        question, q_length, key = question.to(device), q_length.to(device), key.to(device)
        key_num_length, key_word_length, value = key_num_length.to(device), key_word_length.to(device), value.to(device)
        candidate, cand_length, answer = candidate.to(device), cand_length.to(device), answer.to(device)
        index = model.predict(question, q_length, key, key_num_length,
                              key_word_length, value, candidate, cand_length)
        total += question.shape[0]
        correct += (index == answer).sum()
    return float(correct) / total

model = KVMemoryReader(config.d_embed, config.n_embed, config.hop)
model = model.to(device)
# here lr is divide by batch size since loss is accumulated 
optimizer = optim.Adam(model.parameters(), lr=config.lr)
print("Training setting: lr {0}, batch size {1}".format(config.lr, config.batch_size))

loss_function = nn.NLLLoss()

print("{} batch expected".format(len(train_q) * config.epoch / config.batch_size))
print('Getting data ready...')
train_q, train_key, train_value, train_cand, train_a = modify(train_q, train_w, train_e_p, train_a, neg_sampling=True)
dev_q, dev_key, dev_value, dev_cand, dev_a = modify(dev_q, dev_w, dev_e_p, dev_a)
test_q, test_key, test_value, test_cand, test_a = modify(test_q, test_w, test_e_p, list(map(lambda x: torch.tensor(x[:1]), test_a)))

train_q_word_lengths, train_key_num_lengths, train_key_word_lengths, train_cand_lengths = get_data_lengths(
    train_q, train_key, train_cand
)
dev_q_word_lengths, dev_key_num_lengths, dev_key_word_lengths, dev_cand_lengths = get_data_lengths(
    dev_q, dev_key, dev_cand    
)
test_q_word_lengths, test_key_num_lengths, test_key_word_lengths, test_cand_lengths = get_data_lengths(
    test_q, test_key, test_cand
)

zipped_data = list(zip(
    train_q, train_q_word_lengths, train_key, train_key_num_lengths, train_key_word_lengths,
    train_value, train_cand, train_cand_lengths, train_a
))
train_dataloader = DataLoader(
    zipped_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=pad_batch
)

zipped_data = list(zip(
    dev_q, dev_q_word_lengths, dev_key, dev_key_num_lengths, dev_key_word_lengths,
    dev_value, dev_cand, dev_cand_lengths, dev_a
))
dev_dataloader = DataLoader(
    zipped_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=pad_batch
)

zipped_data = list(zip(
    test_q, test_q_word_lengths, test_key, test_key_num_lengths, test_key_word_lengths,
    test_value, test_cand, test_cand_lengths, test_a
))
test_dataloader = DataLoader(
    zipped_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=pad_batch
)

train(config.epoch)
save_model(config.epoch)
