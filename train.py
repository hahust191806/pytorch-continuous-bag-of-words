from model import CBOW
from utils import load_dataset
import torch 
import torchtext 
import os 
import collections 
import random 
import numpy as np 
import builtins 

# select device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, test_dataset, _, vocab, tokenizer = load_dataset()

# init model 

# encode tokenizer to index 
def encode(x, vocabulary, tokenizer = tokenizer):
    return [vocabulary[s] for s in tokenizer(x)]

# get lengh of vocabulary 
vocab_size = len(vocab)

model = CBOW(vocab_size=vocab_size).to(device)

def to_cbow(sent, window_size=2): 
    res = []
    for i, x in enumerate(sent):
        for j in range(max(0, i-window_size), min(i+window_size+1, len(sent))):
            if i!= j: 
                res.append([sent[j], x])
    return res 

X = []
Y = []
for i, x in zip(range(10000), train_dataset):
    for w1, w2 in to_cbow(encode(x[1], vocab), window_size = 5):
        X.append(w1)
        Y.append(w2)

# convert numpy objectives to tensors objectives 
X = torch.tensor(X)
Y = torch.tensor(Y)

# define class to iterate the datasets 
class SimpleIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, X, Y):
        super(SimpleIterableDataset).__init__()
        self.data = []
        for i in range(len(X)):
            self.data.append( (Y[i], X[i]) )
        random.shuffle(self.data)

    def __iter__(self):
        return iter(self.data)
    
ds = SimpleIterableDataset(X, Y)
dl = torch.utils.data.DataLoader(ds, batch_size = 256)

def train_epoch(net, dataloader, lr = 0.01, optimizer = None, loss_fn = torch.nn.CrossEntropyLoss(), epochs = None, report_freq = 1):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr = lr)
    loss_fn = loss_fn.to(device)
    net.train()

    for i in range(epochs):
        total_loss, j = 0, 0, 
        for labels, features in dataloader:
            optimizer.zero_grad()
            features, labels = features.to(device), labels.to(device)
            out = net(features)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
            j += 1
        if i % report_freq == 0:
            print(f"Epoch: {i+1}: loss={total_loss.item()/j}")

    return total_loss.item()/j


train_epoch(net = model, dataloader = dl, optimizer = torch.optim.SGD(model.parameters(), lr = 0.1), loss_fn = torch.nn.CrossEntropyLoss(), epochs = 10)