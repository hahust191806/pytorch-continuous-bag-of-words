import torch 
import torch.nn as nn
# import torchtext 
import os 
import collections 
import random 
import numpy as np 

# select device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defined model 
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size=30) -> None:
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_size)
        self.linear = nn.Linear(in_features = embedding_size, out_features = vocab_size)
        
    def forward(self, x) -> torch.Tensor: 
        return self.linear(self.embedder(x))
    
model = CBOW(1000, 30).to(device)

# print(model)