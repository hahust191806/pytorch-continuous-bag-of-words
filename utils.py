import torchtext 
import numpy as np 
import random 
import builtins 
import os 
import collections 

def load_dataset(ngrams = 1, min_freq = 1, vocab_size = 5000, lines_cnt = 500):
    # generate tokenizer function for a string sentence 
    # "You can now install TorchText using pip!" ==> 'you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!'
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    print("Loading dataset...")
    test_dataset, train_dataset = torchtext.datasets.AG_NEWS(root='./data')
    train_dataset = list(train_dataset)
    test_dataset = list(test_dataset)
    classes = ['World', 'Sports', 'Business', 'Sci/Tech']
    print('Building vocab...')
    counter = collections.Counter()
    for i, (_, line) in enumerate(train_dataset): 
        counter.update(torchtext.data.utils.ngrams_iterator(tokenizer(line), ngrams=ngrams))
        if i == lines_cnt: 
            break 
    vocab = torchtext.vocab.Vocab(collections.Counter(dict(counter.most_common(vocab_size))))
    return train_dataset, test_dataset, classes, vocab, tokenizer 

# test 
# train_dataset, test_dataset, _, vocab, tokenizer = load_dataset()

# print(_)
# print(vocab)
