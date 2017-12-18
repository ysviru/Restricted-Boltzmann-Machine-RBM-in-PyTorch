from __future__ import division

from BernoulliRBMBase import BernoulliRBMBase

import torch
import torch.utils.data
import time
from tqdm import tqdm
import sys
import math


class BernoulliRBM(BernoulliRBMBase):
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
        super(BernoulliRBM, self).__init__(n_visible_units,
                                           n_hidden_units,
                                           init_weight_variance,
                                           learning_rate,
                                           n_epochs,
                                           batch_size,
                                           verbose,
                                           xavier_init,
                                           learning_rate_decay,
                                           increase_to_cd_k,
                                           k)
        self.desc = "BernoulliRBM"

    def _fit(self, batch_matrix, epoch):
        if self.increase_to_cd_k:
            # Start with CD_1 and slowly increase to CD_k
            n_gibbs_sampling_steps = int(math.ceil((epoch / self.n_epochs) * self.k))
        else:
            n_gibbs_sampling_steps = self.k

        if self.learning_rate_decay:
            lr = self.learning_rate / epoch
        else:
            lr = self.learning_rate

        # Positive phase
        positive_hidden_prob = self.get_prob_hiddens(batch_matrix)
        positive_hidden_activations = positive_hidden_prob.bernoulli()
        positive_statistics = torch.matmul(positive_hidden_activations, batch_matrix.t())

        # Negative phase, Gibbs sampling
        hidden_activations = positive_hidden_activations
        for _ in range(n_gibbs_sampling_steps):
            prob_v_ = self.get_prob_visibles(hidden_activations)
            prob_h_ = self.get_prob_hiddens(prob_v_)
            hidden_activations = prob_h_.bernoulli()
        negative_visible_prob = prob_v_
        negative_hidden_prob = prob_h_
        negative_statistics = torch.matmul(negative_hidden_prob, negative_visible_prob.t())

        # Compute gradient and bias updates
        g = positive_statistics - negative_statistics
        grad_update = g / self.batch_size

        v_bias_update = torch.sum(batch_matrix - negative_visible_prob, dim=1) / self.batch_size
        h_bias_update = torch.sum(positive_hidden_prob - negative_hidden_prob, dim=1) / self.batch_size

        self.W += lr * grad_update
        self.v_bias += lr * v_bias_update
        self.h_bias += lr * h_bias_update

        cost_ = torch.sum((batch_matrix - negative_visible_prob) ** 2, dim=0)

        return torch.mean(cost_), torch.sum(torch.abs(grad_update))

    # Fit RBM to training data
    def fit(self,
            training_samples_matrix):
        batches = self.get_batches(training_samples_matrix)
        begin = time.time()
        for epoch in range(1, self.n_epochs + 1):
            cost_ = torch.FloatTensor(len(batches), 1)
            grad_ = torch.FloatTensor(len(batches), 1)
            for i in tqdm(range(len(batches)), ascii=True, desc="(" + self.desc + ", fitting)", file=sys.stdout):
                cost_[i], grad_[i] = self._fit(batches[i], epoch)

            if self.verbose:
                end = time.time()
                print("epoch ", epoch,
                      ", avg_cost = ", torch.mean(cost_),
                      ", std_cost = ", torch.std(cost_),
                      ", avg_grad = ", torch.mean(grad_),
                      ", std_grad = ", torch.std(grad_),
                      ", time elapsed = ", end - begin)
                begin = end
