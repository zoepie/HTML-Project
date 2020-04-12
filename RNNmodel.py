import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.optim as optim
import json


class ProjectDataset(Dataset):
    def __init__(self, split):
        if split == 'train':
            self.len = 23829

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.from_numpy(np.load('dataset/tmp/%d/%d.npy' % ((index // 1000), index)))


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
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, nonlinearity='tanh', batch_first=True)
        self.prob = nn.Sequential(nn.Linear(hidden_dim, vocab_size),
                                  nn.LogSoftmax(dim=-1))
        self.crit = nn.NLLLoss(ignore_index=793, reduction='sum')

    def forward(self, sentence):
        batch_size = sentence.size(0)
        embeds = self.word_embeddings(sentence)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=sentence.device)
        out, _ = self.rnn(embeds, h_0)
        out = self.prob(out)
        return out

    def loss(self, x, y):
        return self.crit(x.transpose(1, 2), y)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def train():
    dataset = ProjectDataset('train')
    train_loader = DataLoader(dataset,
                              batch_size=1024,
                              collate_fn=PackSequence(),
                              shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN(512, 1024, 794, 2).to(device)
    opt = optim.SGD(model.parameters(), lr=0.001)

    epoch = 50
    for e in range(1, epoch):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.long().to(device)
            opt.zero_grad()

            output = model(batch[:, :-1])
            loss = model.loss(output, batch[:, 1:])
            loss.backward()
            opt.step()
            norm = (batch != 793).sum().item()
            acc = torch.eq(output.max(-1)[1], batch[:, 1:]).sum().item() / norm * 100.
            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tACC: {:.2f}\tLoss: {:.6f}'.format(
                    e,
                    i * len(batch),
                    len(train_loader.dataset),
                    100. * i / len(train_loader),
                    acc,
                    loss.item() / norm))


if __name__ == '__main__':
    train()
