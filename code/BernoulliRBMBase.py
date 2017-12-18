from __future__ import division

import torch
import torch.utils.data
import _pickle as pickle


class BernoulliRBMBase(object):
    def __init__(self,
                 n_visible_units,
                 n_hidden_units,
                 init_weight_variance=0.01,
                 learning_rate=0.0001,
                 n_epochs=5,
                 batch_size=32,
                 verbose=False,
                 xavier_init=False,
                 learning_rate_decay=False,
                 increase_to_cd_k=False,
                 k=5):
        self.n_visible_units = n_visible_units
        self.n_hidden_units = n_hidden_units
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.xavier_init = xavier_init
        self.learning_rate_decay = learning_rate_decay
        self.increase_to_cd_k = increase_to_cd_k
        self.k = k
        self.xavier_value = torch.sqrt(torch.FloatTensor([1.0 / (self.n_visible_units + self.n_hidden_units)]))
        # Weight matrix
        if not self.xavier_init:
            self.W = torch.randn(self.n_hidden_units, self.n_visible_units) * init_weight_variance
        else:
            self.W = -self.xavier_value \
                     + torch.rand(self.n_hidden_units, self.n_visible_units) * (2 * self.xavier_value)
        # Biases
        self.v_bias = torch.zeros(n_visible_units, 1)
        self.h_bias = torch.zeros(n_hidden_units, 1)

        self.desc = "RBM"

    # p(v=1|h)
    def get_prob_visibles(self, hidden_state):
        if len(hidden_state.shape) == 1:
            temp = torch.matmul(self.W.t(), hidden_state).view(self.n_visible_units, 1) + self.v_bias
        else:
            num_cols = len(hidden_state[0])
            temp = torch.matmul(self.W.t(), hidden_state) + self.v_bias.repeat(1, num_cols)
        return torch.sigmoid(temp)

    # p(h=1|v)
    def get_prob_hiddens(self, visible_state):
        if len(visible_state.shape) == 1:
            temp = torch.matmul(self.W, visible_state).view(self.n_hidden_units, 1) + self.h_bias
        else:
            num_cols = len(visible_state[0])
            temp = torch.matmul(self.W, visible_state) + self.h_bias.repeat(1, num_cols)
        return torch.sigmoid(temp)

    # Bernoulli draw for visible state given hidden state
    def sample_visibles(self, hidden_state):
        p = self.get_prob_visibles(hidden_state)
        return p, torch.bernoulli(p)

    # Bernoulli draw for hidden state given visible state
    def sample_hiddens(self, visible_state):
        p = self.get_prob_hiddens(visible_state)
        return p, torch.bernoulli(p)

    # Divide training samples into batches
    def get_batches(self, samples):
        return [samples[:, i:i + self.batch_size] for i in range(0, len(samples[0, :]), self.batch_size)]

    # Reconstruct using gibbs sampling.
    def reconstruct(self, sample, n_gibbs=1):
        v = sample
        # Gibbs sampling
        for _ in range(n_gibbs):
            prob_h_ = self.get_prob_hiddens(v)
            prob_v_ = self.get_prob_visibles(prob_h_)
            v = prob_v_.bernoulli()
        return prob_v_, v

    # Helper routines to store and load the model.
    def store_model(self, file_path):
        data_dict = {'weights': self.W, 'v_bias': self.v_bias, 'h_bias': self.h_bias}
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.W = data_dict['weights']
        self.v_bias = data_dict['v_bias']
        self.h_bias = data_dict['h_bias']
