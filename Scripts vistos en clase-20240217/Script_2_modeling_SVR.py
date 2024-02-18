# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

X = housing.data
y = housing.target

miEscalador = MinMaxScaler()

miEscalador.fit(X)

Xesc = miEscalador.transform(X)

# =============================================================================
# Mi primer modelo predictivo!!!
# =============================================================================

from sklearn.svm import SVR

from sklearn.model_selection import ShuffleSplit

sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):

    Xtrain, ytrain = Xesc[train_index], y[train_index]
    Xtest, ytest = Xesc[test_index], y[test_index]

miParamGrid = {"kernel": ["rbf"], "epsilon": [0.1, 0.3], "C": [0.1, 1]}

from sklearn.model_selection import GridSearchCV

miGSCV = GridSearchCV(SVR(), miParamGrid, scoring="r2", cv=10, verbose=3, n_jobs=1)

miGSCV.fit(Xtrain, ytrain)

# =============================================================================
# YA TENGO LA MEJOR CONFIGURACION
# =============================================================================

miMejorModelo = miGSCV.best_estimator_

miMejorModelo.fit(Xtrain, ytrain)

ypred = miMejorModelo.predict(Xtest)

from sklearn.metrics import r2_score

print(r2_score(ytest, ypred))

"""
# =============================================================================
# Regiones de decision
# =============================================================================

from sklearn.decomposition import PCA

miPCA = PCA(n_components=2)

XtrainPCA = Xtrain#miPCA.fit_transform(Xtrain)
Xpcatest = Xtest#miPCA.transform(Xtest)

x0_min, x0_max = Xpcatest[:, 0].min(), Xpcatest[:, 0].max()
x1_min, x1_max = Xpcatest[:, 1].min(), Xpcatest[:, 1].max()

xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.005), 
                            np.arange(x1_min, x1_max, 0.005))

Xtest_all = np.c_[xx0.ravel(), xx1.ravel()]

Xtest_all_invPCA = Xtest_all#miPCA.inverse_transform(Xtest_all)

ypred_all = miMejorModelo.predict(Xtest_all_invPCA)

fig = plt.figure()
ax = fig.add_subplot()

COLORS = ['r','y']
COLORS_TRAINING = ['m','b']

for c in [0,1]:
    indicesPlot = np.where(ypred_all==c)
    ax.scatter(Xtest_all[indicesPlot,0],Xtest_all[indicesPlot,1],c=[COLORS[c] for _ in range(len(indicesPlot))],s=50)
    indicesPlot = np.where(ytrain==c)
    ax.scatter(Xtrain[indicesPlot,0],Xtrain[indicesPlot,1],c=[COLORS_TRAINING[c] for _ in range(len(indicesPlot))],s=150)

"""
