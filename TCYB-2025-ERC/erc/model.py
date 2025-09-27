# -*- coding: utf-8 -*-
"""Evolutionary Regressor Chains Models."""
from random import choices

import torch
import torch.nn.functional as F

from .utils import TrainStreams, BaseLearnerTrain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
rho = 0.1
qmin = 0.016
qmax = 0.875


def generate_new_order(Q):
    """Generate a new order based on Q table."""
    m, _ = Q.shape
    new_order = torch.zeros(m, dtype=torch.int64)
    chosen_list = [False for _ in range(m)]
    for i in range(m):
        if i == 0:
            p = torch.Tensor([Q[j, j].item() for j in range(m)])
            p /= torch.sum(p)
            new_order[i] = choices(list(range(m)), p)[0]
            chosen_list[new_order[i]] = True
        else:
            prev_i = new_order[i - 1]
            p = torch.Tensor([Q[prev_i, j].item() for j in range(m)])
            p[chosen_list] = 0
            p /= torch.sum(p)
            new_order[i] = choices(list(range(m)), p)[0]
            chosen_list[new_order[i]] = True
    return new_order


def update_Q(Q, order, rho=rho, q1=qmin, q2=qmax):
    """Update Q table."""
    m, _ = Q.shape
    flags = torch.zeros((m, m))
    for i, sid in enumerate(order):
        if i == 0:
            flags[sid, sid] = 1
        else:
            prev_sid = order[i - 1]
            flags[prev_sid, sid] = 1
    for i in range(m):
        for j in range(m):
            if flags[i, j]:
                qq = Q[i, j] * (1 - rho) + rho
                Q[i, j] = qq if qq.item() < q2 else q2
            else:
                qq = Q[i, j] * (1 - rho)
                Q[i, j] = qq if qq.item() > q1 else q1


class EvolutionaryRegressorChain:
    """Evolutionary Regressor Chain."""

    def __init__(self, L, learners, Q, device=device):
        """Initialize a EvolutionaryRegressorChain."""
        self.order = torch.randperm(L)
        self.learners = learners
        self.Q = Q
        self.device = device

    def predict(self, x, with_order=None):
        """Make predictions."""
        m, d = x.shape
        if with_order is None:
            with_order = self.order
        y_hat = torch.zeros(m, device=self.device)
        for i, sid in enumerate(with_order):
            if i == 0:
                xx = x[sid, :]
                xy = torch.ones((1, 1), device=self.device) * 0
                y_hat[sid] = self.learners[sid][sid](xx, xy)
            else:
                prev_sid = with_order[i - 1]
                xx = x[sid, :]
                xy = torch.ones((1, 1), device=self.device) * \
                    y_hat[prev_sid].item()
                y_hat[sid] = self.learners[prev_sid][sid](xx, xy)
        return y_hat

    def step(self, x, y, lr=1e-4, weight_decay=1):
        """Update the model and return the predictions."""
        m, d = x.shape
        # predicting and calculate loss
        y_hat1 = self.predict(x)
        loss1 = F.mse_loss(y_hat1, y, reduction='sum')
        order2 = generate_new_order(self.Q)
        y_hat2 = self.predict(x, with_order=order2)
        loss2 = F.mse_loss(y_hat2, y, reduction='sum')
        # update chain order
        if loss1 > loss2:
            self.order = order2
            y_hat = y_hat2
            loss = loss2
        else:
            y_hat = y_hat1
            loss = loss1
        # backward
        for i, sid in enumerate(self.order):
            if i == 0:
                self.learners[sid][sid].zero_grad()
                for para in self.learners[sid][sid].parameters():
                    loss += para.norm(p=2) * weight_decay
            else:
                prev_sid = self.order[i - 1]
                self.learners[prev_sid][sid].zero_grad()
                for para in self.learners[prev_sid][sid].parameters():
                    loss += para.norm(p=2) * weight_decay
        loss.backward()
        # gradient descent
        for i, sid in enumerate(self.order):
            if i == 0:
                paras = self.learners[sid][sid].parameters()
            else:
                prev_sid = self.order[i - 1]
                paras = self.learners[prev_sid][sid].parameters()
            for para in paras:
                para.data.add_(para.grad.data, alpha=-lr)
        # update Q
        update_Q(self.Q, self.order)
        return y_hat


def pruning(y, nn, device=device):
    """Prune and decrease the number of chains."""
    n, m = y.shape
    diversity = torch.zeros((n, n), device=device)
    chosen_sid = torch.zeros(nn, dtype=torch.long, device=device) - 1
    for i in range(n):
        for j in range(n):
            if i != j:
                diversity[i, j] = F.mse_loss(y[i, :], y[j, :])
    maxid = torch.argmax(diversity)
    chosen_sid[0] = maxid.item() // n
    chosen_sid[1] = maxid.item() % n
    for i in range(2, nn):
        diversity_i = torch.zeros(n, device=device)
        for j in range(n):
            if j in chosen_sid:
                continue
            for k in range(i):
                diversity_i[j] += diversity[j, k]
        maxid = torch.argmax(diversity_i)
        chosen_sid[i] = maxid.item()
    return chosen_sid


class EvolutionaryRegressorChains:
    """Evolutional Regressor Chains."""

    def __init__(self, n_chains, L, baselearner, device=device):
        """Initialize EvolutionalRegressorChains."""
        self.n_chains = n_chains
        self.learners = \
            [[baselearner(device=device) for _ in range(L)] for _ in range(L)]
        self.Q = torch.ones((L, L), device=device)
        self.device = device
        self.chains = [EvolutionaryRegressorChain(L,
                                                  self.learners,
                                                  self.Q,
                                                  device=self.device)
                       for _ in range(n_chains)]

    def step(self, x, y, lr=1e-4, weight_decay=1, n_pruning=None):
        """Update the models and return the predictions."""
        m, d = x.shape
        y_hat = torch.zeros((self.n_chains, m), device=self.device)
        for i, chain in enumerate(self.chains):
            y_hat[i, :] = chain.step(x, y, lr, weight_decay)
        if n_pruning is not None:
            sid_after_pruning = pruning(y_hat, n_pruning, self.device)
            y_hat2 = y_hat[sid_after_pruning, :].sum(axis=0) / n_pruning
        else:
            y_hat2 = y_hat.sum(axis=0) / self.n_chains
        loss = F.mse_loss(y_hat2, y, reduction='none')
        return loss


if __name__ == "__main__":
    print('Test function generate_new_order...')
    Q = torch.rand((5, 5), device=device)
    for i in range(5):
        norder = generate_new_order(Q)
        print('  ith order: ', norder)
    print("Test complete.")
    print()
    ###########
    #  split  #
    ###########
    print('Test update_Q...')
    print('Old Q table: ', Q)
    update_Q(Q, norder)
    print('New Q table: ', Q)
    print("Test complete.")
    print()
    ###########
    #  split  #
    ###########
    print('Test ERC on TrainStreams...')
    print('loading data...')
    sydtrain = TrainStreams(device)
    erc_train = EvolutionaryRegressorChains(10, 8, BaseLearnerTrain)
    for i, data in enumerate(sydtrain):
        if i == 10:
            break
        print('  Processing sample {}...'.format(i + 1))
        x, y = data
        erc_train.step(x, y)
    print("Test complete.")
