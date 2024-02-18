# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import make_classification

X, y = make_classification(
    1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
)

miEscalador = MinMaxScaler()

miEscalador.fit(X)

Xesc = miEscalador.transform(X)

# =============================================================================
# Mi primer modelo predictivo!!!
# =============================================================================

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):

    Xtrain, ytrain = Xesc[train_index], y[train_index]
    Xtest, ytest = Xesc[test_index], y[test_index]

miParamGrid = {"kernel": ["rbf"], "gamma": [1e5], "C": [1000]}

from sklearn.model_selection import GridSearchCV

miGSCV = GridSearchCV(
    SVC(), miParamGrid, scoring="accuracy", cv=10, verbose=2, n_jobs=-1
)

miGSCV.fit(Xtrain, ytrain)

# =============================================================================
# YA TENGO LA MEJOR CONFIGURACION
# =============================================================================

miMejorModelo = miGSCV.best_estimator_

miMejorModelo.fit(Xtrain, ytrain)

ypred = miMejorModelo.predict(Xtest)

from sklearn.metrics import accuracy_score

print(accuracy_score(ytest, ypred))

# =============================================================================
# Regiones de decision
# =============================================================================

from sklearn.decomposition import PCA

miPCA = PCA(n_components=2)

XtrainPCA = Xtrain  # miPCA.fit_transform(Xtrain)
Xpcatest = Xtest  # miPCA.transform(Xtest)

x0_min, x0_max = Xpcatest[:, 0].min(), Xpcatest[:, 0].max()
x1_min, x1_max = Xpcatest[:, 1].min(), Xpcatest[:, 1].max()

xx0, xx1 = np.meshgrid(
    np.arange(x0_min, x0_max, 0.005), np.arange(x1_min, x1_max, 0.005)
)

Xtest_all = np.c_[xx0.ravel(), xx1.ravel()]

Xtest_all_invPCA = Xtest_all  # miPCA.inverse_transform(Xtest_all)

ypred_all = miMejorModelo.predict(Xtest_all_invPCA)

fig = plt.figure()
ax = fig.add_subplot()

COLORS = ["r", "y"]
COLORS_TRAINING = ["m", "b"]

for c in [0, 1]:
    indicesPlot = np.where(ypred_all == c)
    ax.scatter(
        Xtest_all[indicesPlot, 0],
        Xtest_all[indicesPlot, 1],
        c=[COLORS[c] for _ in range(len(indicesPlot))],
        s=50,
    )
    indicesPlot = np.where(ytrain == c)
    ax.scatter(
        Xtrain[indicesPlot, 0],
        Xtrain[indicesPlot, 1],
        c=[COLORS_TRAINING[c] for _ in range(len(indicesPlot))],
        s=150,
    )
