from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# LEo de disco
# =============================================================================

fileName = "./digits.csv"

fid = open(fileName, "r")
X = []
y = []

for line in fid:
    lineParsed = line.strip().split(",")
    X.append(list(map(int, lineParsed[:-1])))
    y.append(int(lineParsed[-1]))

X = np.array(X)
y = np.array(y)

# =============================================================================
# Particiono
# =============================================================================

from sklearn.model_selection import train_test_split

Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.3, stratify=y)

# =============================================================================
# Normalizo
# =============================================================================

miScaler = MinMaxScaler()
miScaler.fit(Xtr)
Xtr_norm = miScaler.transform(Xtr)
Xtest_norm = miScaler.transform(Xtest)
Xtest_recovered = miScaler.inverse_transform(Xtest_norm)

# =============================================================================
# MODELO
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

miModelo = KNeighborsClassifier()

miModelo.fit(Xtr_norm, ytr)

ypred = miModelo.predict(Xtest_norm)

print(accuracy_score(ytest, ypred))

from sklearn.model_selection import GridSearchCV

misParametros = {
    "n_neighbors": [3, 5, 7, 9],
    "p": [1, 2, 3, 4, 5],
    "weights": ["uniform", "distance"],
}

miGSCV = GridSearchCV(
    estimator=miModelo, param_grid=misParametros, scoring="accuracy", cv=5, verbose=1
)

miGSCV.fit(Xtr_norm, ytr)
miMejorModelo = miGSCV.best_estimator_

miMejorModelo.fit(Xtr_norm, ytr)
ypred = miMejorModelo.predict(Xtest_norm)

print(accuracy_score(ytest, ypred))


# =============================================================================
# Pintar puntos test: reduzco dimensionalidad y pinto labels
# =============================================================================

miPCA = PCA(n_components=3)
from mpl_toolkits.mplot3d import Axes3D

miPCA.fit(Xtr)
Xtr_norm_pca = miPCA.fit_transform(Xtr_norm)
Xtest_norm_pca = miPCA.transform(Xtest_norm)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    Xtest_norm_pca[:, 0], Xtest_norm_pca[:, 1], Xtest_norm_pca[:, 2], c=ypred, s=50
)

"""
# =============================================================================
# Proyectar con otro método (tSNE) [OPCIONAL]
# =============================================================================

from sklearn.manifold import TSNE

miTSNE = TSNE(n_components=3)

Xtest_norm_tsne = miTSNE.fit_transform(Xtest_norm)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xtest_norm_tsne[:,0],Xtest_norm_tsne[:,1],Xtest_norm_tsne[:,2],c=ypred,s=50)

# =============================================================================
# Ahora usamos el inverse PCA sobre el espacio muestreado para obtener las labels
# =============================================================================

# MUESTREAMOS

x0_min, x0_max = Xtest_norm_pca[:, 0].min(), Xtest_norm_pca[:, 0].max()
x1_min, x1_max = Xtest_norm_pca[:, 1].min(), Xtest_norm_pca[:, 1].max()
x2_min, x2_max = Xtest_norm_pca[:, 2].min(), Xtest_norm_pca[:, 2].max()

xx0, xx1, xx2 = np.meshgrid(np.arange(x0_min, x0_max, 0.2), 
                            np.arange(x1_min, x1_max, 0.2),
                            np.arange(x2_min, x2_max, 0.2))

Xtest_all = np.c_[xx0.ravel(), xx1.ravel(), xx2.ravel()]

Xtest_all_invPCA = miPCA.inverse_transform(Xtest_all)

ypred_all = miMejorModelo.predict(Xtest_all_invPCA)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xtest_all[:,0],Xtest_all[:,1],Xtest_all[:,2],cmap = 'tab10',c=ypred_all,s=50,alpha=0.5)


miPCA = PCA(n_components = 2)

miPCA.fit(Xtr)
Xtr_norm_pca = miPCA.fit_transform(Xtr_norm)
Xtest_norm_pca = miPCA.transform(Xtest_norm)

x0_min, x0_max = Xtest_norm_pca[:, 0].min(), Xtest_norm_pca[:, 0].max()
x1_min, x1_max = Xtest_norm_pca[:, 1].min(), Xtest_norm_pca[:, 1].max()

xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.02), 
                            np.arange(x1_min, x1_max, 0.02))

Xtest_all = np.c_[xx0.ravel(), xx1.ravel()]

Xtest_all_invPCA = miPCA.inverse_transform(Xtest_all)

ypred_all = miMejorModelo.predict(Xtest_all_invPCA)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(Xtest_all[:,0],Xtest_all[:,1],cmap = 'tab10',c=ypred_all,s=50,alpha=0.5)
















"""
"""
x0_min, x0_max = Xtest_std_pca[:, 0].min(), Xtest_std_pca[:, 0].max()
x1_min, x1_max = Xtest_std_pca[:, 1].min(), Xtest_std_pca[:, 1].max()

xx0, xx1, = np.meshgrid(np.arange(x0_min, x0_max, 0.1), 
                                 np.arange(x1_min, x1_max, 0.1))

Xtest_all = np.c_[xx0.ravel(), xx1.ravel()]

ypred_all = miMejorModelo.predict(Xtest_all)

from sklearn.decomposition import PCA

myPCA = PCA(n_components=2)
myPCA.fit(Xtr)
Xtr_2D = myPCA.transform(Xtr)
Xtest_all_2D = myPCA.transform(Xtest_all)

plt.scatter(Xtest_all_2D[:,0],Xtest_all_2D[:,1],c=ypred_all,s=50)
# ypred_all_ = ypred_all.reshape(xx0.shape)
# plt.contourf(xx, yy, Z, alpha=0.4)
"""
