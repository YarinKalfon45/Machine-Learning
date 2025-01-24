import numpy as np
from torch import nn as nn
import torch.nn.functional
def convert_words_to_indices(sents,vocab_stoi): # 10% grade
    """
    This function takes a list of sentences
    input: list of list of words [[word,word,..,word],..,[word,..,word]] where each word is a string with no spaces
    and returns a new list with the same structure, but where each word is replaced by its index in `vocab_stoi`.
    output: list of lists of integers [[int,int,..,int],..,[int,..,int]] where each int is the idx of the word according to vocab_stoi

    Example:
    >>> convert_words_to_indices([['one', 'in', 'five', 'are', 'over', 'here'], ['other', 'one', 'since', 'yesterday'], ['you']])
    [[148, 98, 70, 23, 154, 89], [151, 148, 181, 246], [248]]
    """
    words_to_indices = []
    for sent in sents:
      indexes = []
      for word in sent:
        indexes.append(vocab_stoi.get(word, -1))
      words_to_indices.append(indexes)
    return words_to_indices

def generate_4grams(seqs): # 10% grade
    """
    This function takes a list of sentences (list of lists) and returns
    a new list containing the 4-grams (four consequentively occuring words)
    that appear in the sentences. Note that a unique 4-gram can appear multiple
    times, one per each time that the 4-gram appears in the data parameter `seqs`.

    Example:

    >>> generate_4grams([[148, 98, 70, 23, 154, 89], [151, 148, 181, 246], [248]])
    [[148, 98, 70, 23], [98, 70, 23, 154], [70, 23, 154, 89], [151, 148, 181, 246]]
    >>> generate_4grams([[1, 1, 1, 1, 1]])
    [[1, 1, 1, 1], [1, 1, 1, 1]]
    """
    four_grams = []
    for seq in seqs:
      if len(seq) > 3:
        for i in range(len(seq)-3):
          four_gram =[] 
          for j in range(4):
            four_gram.append(seq[i+j])
          four_grams.append(four_gram)
    return four_grams

import numpy as np

def make_onehot(data):
    """
    Convert one batch of data in the index notation into its corresponding onehot
    notation. Works for both 1D and 2D inputs.

    If flatten=True, the function will reshape the one-hot output into a single
    vector for each row of input, useful for combining features.

    input - vector with shape D (1D or 2D)
    output - vector with shape:
        - (D, 250) if flatten=False
        - (D, N * 250) if flatten=True and input is 2D
    """
    data = np.array(data)  # Ensure input is a NumPy array
    
    if data.ndim == 1:  # Handle 1D input
        D = data.shape[0]
        onehot = np.zeros((D, 250))
        for i in range(D):
            onehot[i, data[i]] = 1

    elif data.ndim == 2:  # Handle 2D input
        D, N = data.shape
        onehot = np.zeros((D * N, 250))
        for i in range(D):
            for j in range(N):
                onehot[i * N + j, data[i, j]] = 1

    return onehot


class PyTorchMLP(nn.Module): # 35% grade for each model
    def __init__(self):
        super(PyTorchMLP, self).__init__()
        self.num_hidden = 256 # TODO: choose number of hidden neurons
        self.layer1 = nn.Linear(750, self.num_hidden)
        self.layer2 = nn.Linear(self.num_hidden, 250)
        
    def forward(self, inp):
        inp = inp.reshape([-1, 750])
        hidden = self.layer1(inp)
        hidden = torch.nn.functional.relu(hidden)
        output = self.layer2(hidden)
        return output
        # TODO: complete this function
        # Note that we will be using the nn.CrossEntropyLoss(), which computes the softmax operation internally, as loss criterion