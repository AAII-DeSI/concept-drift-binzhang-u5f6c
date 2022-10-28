# -*- coding: utf-8 -*-
"""Source code of Drift Adaptation via Joint Distribution Alignment.

Source code of Drift Adaptation via Joint Distribution Alignment.
"""
import numpy as np
import scipy.linalg


class DAJDA:
    """DAJDA class."""

    def __init__(self, clf, mu=0.5, lamb=1):
        """__init__ for DAJDA."""
        self.clf = clf
        self.mu = mu
        self.lamb = lamb
        self.A = None
        self.source = None

    def fit(self, x, y):
        """Fit method."""
        if self.source is None:
            self.clf.fit(x, y)
            self.source = (x, y)
            return

        Xs, Ys = self.source
        Xt, Yt = x, y
        X = np.hstack((Xs.T, Xt.T))
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = np.unique(Ys)
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = e * e.T
        N = np.zeros(n)
        for c in C:
            e = np.zeros((n, 1))
            e[np.where(Ys == c)] = 1/len(Ys[np.where(Ys == c)])
            ind = np.where(Yt == c)
            inds = tuple([item + ns for item in ind])
            if len(Yt[np.where(Yt == c)]) == 0:
                e[inds] = 0
            else:
                e[inds] = -1/len(Yt[np.where(Yt == c)])
            N = N + np.dot(e, e.T)

        M = M * self.mu + (1-self.mu) * N
        a = np.linalg.multi_dot([X, M, X.T]) + self.lamb * np.eye(m)
        b = np.linalg.multi_dot([X, H, X.T])
        w, self.A = scipy.linalg.eig(a, b)
        w, self.A = w.real, self.A.real

        Z = np.dot(self.A.T, X)
        y_new = np.hstack((Ys, Yt))
        self.clf.fit(Z.T, y_new)
        self.source = (x, y)
        return

    def predict(self, X):
        """Predict method."""
        X = np.array(X)
        if self.A is None:
            return self.clf.predict(X)
        else:
            Z = np.dot(X, self.A)
            return self.clf.predict(Z)
