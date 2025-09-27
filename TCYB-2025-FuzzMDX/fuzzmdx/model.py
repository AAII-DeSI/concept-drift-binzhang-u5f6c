# -*- coding: utf-8 -*-
"""FuzzMDX Stream handlers."""
import numpy as np
from scipy.stats import mannwhitneyu
import torch
import torch.nn as nn


# rank sum test
def rank_sum_test(memb1, memb2, alpha=0.05):
    """Rank sum test."""
    _, p = mannwhitneyu(memb1, memb2)
    if p < alpha:
        return True
    return False


# membership functions
class SMF:
    """sigmoid membership function."""

    def __init__(self):
        """Initialize a membership function."""
        self.func = nn.Sequential(nn.Linear(1, 1, dtype=torch.float64),
                                  nn.Sigmoid())
        self.optimizer = torch.optim.SGD(self.func.parameters(), lr=1e-2)

    def membership(self, x):
        """Return the membership of a ndarray."""
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        x = x.reshape((-1, 1))
        return self.func(x)

    def fit(self, x1, x2, epochs=5):
        """Estimate the parameters of membership.

        x1, x2: ndarray of size (n, d)
        """
        x1 = torch.from_numpy(x1)
        n1 = x1.shape[0]
        y1 = torch.ones(n1, dtype=torch.float64)
        x2 = torch.from_numpy(x2)
        n2 = x2.shape[0]
        y2 = torch.zeros(n2, dtype=torch.float64)
        shuff = torch.randperm(n1 + n2)
        X = torch.cat((x1, x2))
        X = X[shuff]
        Y = torch.cat((y1, y2))
        Y = Y[shuff]

        for _ in range(epochs):
            for i in range((n1 + n2) // 100):
                i1 = i * 100
                i2 = i1 + 100
                xi = X[i1:i2]
                yi = Y[i1:i2]
                yhat = self.membership(xi).reshape(-1)
                loss = nn.functional.mse_loss(yhat, yi)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


# stream handler, each stream has a learner and a membership fucntion
class StreamHandler:
    """Handle a stream in dynamic environment."""

    def __init__(self, base_learner, random_state=None):
        """Initialize a StreamHandler."""
        self.learner = base_learner
        self.mf = SMF()
        self.hist_memb = None

    def fit(self, x, y, x1, y1, epochs=100, sample_weight=None):
        """Fit the model."""
        self.learner.fit(x, y, sample_weight=sample_weight)
        yhat = self.learner.predict(x)
        loss = (yhat - y) ** 2
        yhat1 = self.learner.predict(x1)
        loss1 = (yhat1 - y1) ** 2
        self.mf.fit(loss, loss1, epochs=epochs)
        self.hist_memb = self.mf.membership(loss).detach().cpu().numpy()

    def partial_fit(self, x, y, x1, y1, epochs=100, sample_weight=None):
        """Partial fit the model."""
        self.learner.partial_fit(x, y, sample_weight=sample_weight)
        yhat = self.learner.predict(x)
        loss = (yhat - y) ** 2
        yhat1 = self.learner.predict(x1)
        loss1 = (yhat1 - y1) ** 2
        self.mf.fit(loss, loss1, epochs=epochs)
        self.hist_memb = self.mf.membership(loss).detach().cpu().numpy()

    def score(self, x, y, return_memb=False):
        """Make predictions and calculate scores.

        x: ndarray of size (n, d)
        y: ndarray of size (n,)
        """
        yhat = self.learner.predict(x)
        loss = (yhat - y) ** 2
        memb = self.mf.membership(loss).detach().cpu().numpy()
        if return_memb:
            return memb
        drift = rank_sum_test(self.hist_memb, memb)
        return loss, drift


if __name__ == "__main__":
    mf = SMF()
    x1 = np.random.normal(1, 1, 1000)
    x2 = np.random.normal(2, 1, 1000)
    mf.fit(x1, x2, 10000)
    print(mf.membership(x1).reshape(-1))
    print(mf.membership(x2).reshape(-1))
