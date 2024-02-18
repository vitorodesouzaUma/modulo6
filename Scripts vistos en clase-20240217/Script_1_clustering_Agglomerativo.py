# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering

# https://stackoverflow.com/questions/26851553/sklearn-agglomerative-clustering-linkage-matrix


def get_distances(X, model, mode="l2"):
    distances = []
    weights = []
    children = model.children_
    dims = (X.shape[1], 1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1 - c2)
        cc = ((c1W * c1) + (c2W * c2)) / (c1W + c2W)

        X = np.vstack((X, cc.T))

        newChild_id = X.shape[0] - 1

        # How to deal with a higher level cluster merge with lower distance:
        if mode == "l2":  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2 + c2Dist**2) ** 0.5
            dNew = (d**2 + added_dist**2) ** 0.5
        elif (
            mode == "max"
        ):  # If the previrous clusters had higher distance, use that one
            dNew = max(d, c1Dist, c2Dist)
        elif mode == "actual":  # Plot the actual distance.
            dNew = d

        wNew = c1W + c2W
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append(wNew)
    return distances, weights


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


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

miAggCluster = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

miAggCluster.fit(Xesc)
distance, weight = get_distances(Xesc, miAggCluster)

linkage_matrix = np.column_stack([miAggCluster.children_, distance, weight]).astype(
    float
)
plt.figure(figsize=(20, 10))
dendrogram(linkage_matrix)
# plot_dendrogram(miAggCluster, truncate_mode="level", p=3)


# =============================================================================
# Proyectar con PCA
# =============================================================================
"""
colorTags = ['k','r','g','b','m','c']

from sklearn.decomposition import PCA

miPCA = PCA(n_components=2)
miPCA.fit(Xesc)
Xpca = miPCA.transform(Xesc)

miDBSCAN_2 = DBSCAN(eps=0.1)
miDBSCAN_2.fit(Xpca)

clusters_2 = miDBSCAN_2.labels_


fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(Xpca[:,0],Xpca[:,1],s=60,c=[colorTags[i+1] for i in clusters_2])

#ax = fig.add_subplot(projection='3d')
#ax.scatter(Xpca[:,0],Xpca[:,1],Xpca[:,2],s=80,cmap='tab20',c=clusters)

# =============================================================================
# Computo accuracy, manual y con libreria
# =============================================================================

print(sum(clusters==colores[desorden])/len(clusters))

from sklearn.metrics import accuracy_score, normalized_mutual_info_score

print(accuracy_score(colores[desorden],clusters))

print(normalized_mutual_info_score(colores[desorden],clusters))

"""
