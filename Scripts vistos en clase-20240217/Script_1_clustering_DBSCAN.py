# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

X, colores = load_iris(return_X_y=True)

# =============================================================================
# Stats
# =============================================================================

print("Medias:", np.mean(X, 0))
print("StdDevs:", np.std(X, 0))

# =============================================================================
# Pintamos las caracteristicas
# =============================================================================
desorden = np.random.permutation(150)
# plt.figure()
# plt.plot(X[desorden,0])
# plt.plot(X[desorden,1])
# plt.plot(X[desorden,2])
# plt.plot(X[desorden,3])

X = X[desorden]

# =============================================================================
# Normalizar
# =============================================================================

miEscalador = MinMaxScaler()

miEscalador.fit(X)

Xesc = miEscalador.transform(X)

# =============================================================================
# Clustering con Kmeans
# =============================================================================

from sklearn.cluster import DBSCAN

miDBSCAN = DBSCAN(eps=0.1)

miDBSCAN.fit(Xesc)

clusters = miDBSCAN.labels_

# =============================================================================
# Proyectar con PCA
# =============================================================================

colorTags = ["k", "r", "g", "b", "m", "c"]

from sklearn.decomposition import PCA

miPCA = PCA(n_components=2)
miPCA.fit(Xesc)
Xpca = miPCA.transform(Xesc)

miDBSCAN_2 = DBSCAN(eps=0.1)
miDBSCAN_2.fit(Xpca)

clusters_2 = miDBSCAN_2.labels_


fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(Xpca[:, 0], Xpca[:, 1], s=60, c=[colorTags[i + 1] for i in clusters_2])

# ax = fig.add_subplot(projection='3d')
# ax.scatter(Xpca[:,0],Xpca[:,1],Xpca[:,2],s=80,cmap='tab20',c=clusters)

# =============================================================================
# Computo accuracy, manual y con libreria
# =============================================================================

print(sum(clusters == colores[desorden]) / len(clusters))

from sklearn.metrics import accuracy_score, normalized_mutual_info_score

print(accuracy_score(colores[desorden], clusters))

print(normalized_mutual_info_score(colores[desorden], clusters))
