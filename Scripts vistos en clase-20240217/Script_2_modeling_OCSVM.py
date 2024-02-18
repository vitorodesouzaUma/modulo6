# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import make_blobs

X, y_basura = make_blobs(1000, n_features=2, centers=4, random_state=13)

miEscalador = MinMaxScaler()

miEscalador.fit(X)

Xtrain = miEscalador.transform(X)

# =============================================================================
# Mi primer modelo predictivo!!!
# =============================================================================

from sklearn.svm import OneClassSVM

miParamGrid = {"kernel": ["rbf"], "epsilon": [0.1, 0.3], "C": [0.1, 1]}

miModelo = OneClassSVM(nu=0.5)
miModelo.fit(Xtrain)

# =============================================================================
# Regiones de decision
# =============================================================================


x0_min, x0_max = Xtrain[:, 0].min(), Xtrain[:, 0].max()
x1_min, x1_max = Xtrain[:, 1].min(), Xtrain[:, 1].max()

xx0, xx1 = np.meshgrid(
    np.arange(x0_min, x0_max, 0.005), np.arange(x1_min, x1_max, 0.005)
)

Xtrain_all = np.c_[xx0.ravel(), xx1.ravel()]

ypred_all = ((miModelo.predict(Xtrain_all) + 1) / 2.0).astype(int)

fig = plt.figure()
ax = fig.add_subplot()

COLORS = ["r", "g"]

for c in [0, 1]:
    indicesPlot = np.where(ypred_all == c)[0]
    ax.scatter(
        Xtrain_all[indicesPlot, 0],
        Xtrain_all[indicesPlot, 1],
        c=[COLORS[c] for _ in range(len(indicesPlot))],
        s=50,
    )
    ax.scatter(Xtrain[:, 0], Xtrain[:, 1], c=["k" for _ in range(len(Xtrain))], s=150)
