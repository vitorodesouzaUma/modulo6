# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_digits
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

X, y = load_digits(return_X_y=True)

miEscalador = MinMaxScaler()

miEscalador.fit(X)

Xesc = miEscalador.transform(X)

# =============================================================================
# Mi primer modelo predictivo!!!
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):

    Xtrain, ytrain = Xesc[train_index], y[train_index]
    Xtest, ytest = Xesc[test_index], y[test_index]

miK = 5
"""
from sklearn.neighbors import KNeighborsClassifier

miKNN = KNeighborsClassifier(n_neighbors=miK)

miKNN.fit(Xtrain,ytrain)

ypred = miKNN.predict(Xtest)

from sklearn.metrics import accuracy_score

print(accuracy_score(ytest,ypred))
"""
"""
from sklearn.model_selection import cross_val_score


scores = cross_val_score(KNeighborsClassifier(n_neighbors=miK),Xtrain,ytrain,scoring='accuracy',cv=5)

"""

miParamGrid = {
    "weights": ["uniform", "distance"],
    "metric": ["cityblock", "minkowski"],
    "p": [2, 3, 4, 5, 6, 7],
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
}

from sklearn.model_selection import GridSearchCV

miGSCV = GridSearchCV(
    KNeighborsClassifier(), miParamGrid, scoring="accuracy", cv=10, verbose=2, n_jobs=-1
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


from sklearn.decomposition import PCA

miPCA = PCA(n_components=2)
miPCA.fit(Xesc)
Xpca = miPCA.transform(Xesc)
Xpcatest = Xpca[test_index]

aciertoOno = ["k" if ypred[i] == ytest[i] else "m" for i in range(len(ytest))]

fig = plt.figure()
axs = fig.subplots(1, 2)
axs[0].scatter(Xpcatest[:, 0], Xpcatest[:, 1], cmap="tab10", s=60, c=aciertoOno)
axs[1].scatter(Xpcatest[:, 0], Xpcatest[:, 1], cmap="tab10", s=60, c=ytest)

from sklearn.metrics import confusion_matrix

confusion_matrix(ytest, ypred)

plt.matshow(confusion_matrix(ytest, ypred), cmap="jet")


"""
# =============================================================================
# Stats
# =============================================================================

print('Medias:',np.mean(X,0))
print('StdDevs:',np.std(X,0))

# =============================================================================
# Pintamos las caracteristicas
# =============================================================================
desorden = np.random.permutation(150)
#plt.figure()
#plt.plot(X[desorden,0])
#plt.plot(X[desorden,1])
#plt.plot(X[desorden,2])
#plt.plot(X[desorden,3])

X = X[desorden]

# =============================================================================
# Normalizar
# =============================================================================



# =============================================================================
# Clustering con Kmeans
# =============================================================================

miAggCluster = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

miAggCluster.fit(Xesc)
distance, weight = get_distances(Xesc,miAggCluster)

linkage_matrix = np.column_stack([miAggCluster.children_, distance, weight]).astype(float)
plt.figure(figsize=(20,10))
dendrogram(linkage_matrix)
#plot_dendrogram(miAggCluster, truncate_mode="level", p=3)


# =============================================================================
# Proyectar con PCA
# =============================================================================

colorTags = ['k','r','g','b','m','c']



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
