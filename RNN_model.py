#!/usr/bin/python

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from preprocess import find_child, read_file, get_all_paths
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import itertools


class ProjectDataset(Dataset):
    def __init__(self, split):
        self.split = split
        if split == 'train':
            self.len = 23829
        elif split == 'test':
            self.len = 0
        else:
            raise ValueError('Invalid split name.')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.from_numpy(np.load('dataset/%s/%d/%d.npy' % (self.split, (index // 1000), index)))


class PackSequence:
    def __call__(self, batch):
        batch = sorted(batch,
                       key=lambda x: x.size(0),
                       reverse=True)
        return pad_sequence(batch, padding_value=793).transpose(0, 1)


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.prob = nn.Sequential(nn.Linear(hidden_dim, vocab_size),
                                  nn.LogSoftmax(dim=-1))
        self.crit = nn.NLLLoss(ignore_index=793, reduction='sum')

    def forward(self, sentence):
        batch_size = sentence.size(0)
        embeds = self.word_embeddings(sentence)
        # h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=sentence.device)
        out, _ = self.rnn(embeds)
        out = self.prob(out)
        return out

    def loss(self, x, y):
        return self.crit(x.transpose(1, 2), y)


class ClassNode(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child):
        self.children.append(child)


def draw_result(x, y, x_label, y_label, title):
    plt.plot(x, y, color='green')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(len(x)))
    plt.title(title)
    plt.show()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def test_loader(index):
    with open('name2idx.json', 'r') as fp:
        name2idx = json.load(fp)
    root = ClassNode('root')

    body = read_file('dataset/test/%d.txt' % index).body
    if not body:
        return None
        # raise ValueError('Invalid html file.')
    find_child(body, root)
    path_list = get_all_paths(root)
    output = []

    for j in range(len(path_list)):
        # print('Processing %d-%d...' % (index, j))
        is_looping = True
        for k in itertools.product(*path_list[j][1:]):
            idx_list = []
            for i in k:
                try:
                    idx_list.append(name2idx[i])
                except KeyError:
                    print('Class names out of scope: ', i)
                    is_looping = False
                    break
            if is_looping:
                output.append(torch.tensor(idx_list).unsqueeze(0))
            else:
                break
    return output


def test(checkpoint):
    with open('idx2name.json', 'r') as fp:
        idx2name = json.load(fp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN(512, 1024, 794, 2).to(device)
    model.load_state_dict(torch.load('model_epoch_%d.pt' % checkpoint))
    model.eval()
    for i in range(3):
        test_list = test_loader(i)
        if test_list:
            for t in test_list:
                output = model.forward(t[:, :-1].to(device))
                prob = torch.softmax(output, dim=-1).squeeze(dim=0)
                for idx, token in enumerate(t[0, 1:]):
                    if prob[idx, token.item()].item() * 100 > 1:
                        print('Parent: %s, \t\t\tChild: %s, \t\t\tProb: %f' % (
                            idx2name[str(t[0, idx].item())], idx2name[str(token.item())],
                            prob[idx, token.item()].item() * 100))


def train():
    dataset = ProjectDataset('train')
    train_loader = DataLoader(dataset,
                              batch_size=32,
                              collate_fn=PackSequence(),
                              shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN(512, 1024, 794, 2).to(device)
    # opt = optim.Adam(model.parameters(), lr=0.0001)
    opt = optim.SGD(model.parameters(), lr=0.0001)
    x_SGD = []
    y_SGD = []
    epoch = 10
    for e in range(1, epoch + 1):
        x_SGD.append(e - 1)
        model.train()
        list_acc = []
        for i, batch in enumerate(train_loader):
            batch = batch.long().to(device)
            opt.zero_grad()
            output = model.forward(batch[:, :-1])
            loss = model.loss(output, batch[:, 1:])
            loss.backward()
            opt.step()
            norm = (batch[:, :-1] != 793).sum().item()
            acc = torch.eq(output.max(-1)[1], batch[:, 1:]).sum().item() / norm * 100.
            list_acc.append(acc)
            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tACC: {:.2f}\tLoss: {:.6f}'.format(
                    e,
                    i * len(batch),
                    len(train_loader.dataset),
                    100. * i / len(train_loader),
                    acc,
                    loss.item() / norm))
        ave = sum(list_acc) / len(list_acc)
        y_SGD.append(ave)
        torch.save(model.state_dict(), 'model_epoch_%d.pt' % e)
    print(x_SGD)
    print(y_SGD)
    # draw_result(x, y, "epoch", "percentage of accuracy","512, 1024, 794, 2,ADAM")


def main(mode):
    if mode == 'train':
        print('Training...')
        train()
    elif mode == 'test':
        test(10)


if __name__ == '__main__':
    main(sys.argv[1])
