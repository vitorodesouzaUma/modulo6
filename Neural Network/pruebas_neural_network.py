# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_svmlight_file, load_svmlight_files

X_train, y_train, X_test, y_test = load_svmlight_files(("./data/adult1.svm", "./data/adult1_test.svm"))


from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor

pl = Pipeline([
    ('standard_scaler', StandardScaler(with_mean=False)),
    ('mlpc', MLPClassifier())
])

from sklearn.model_selection import GridSearchCV

param_grid = {'mlpc__hidden_layer_sizes':[(50,),(100,),(10,),(50,50,),(150,150,)],
              'mlpc__activation':['identity', 'logistic', 'tanh', 'relu'],
              'mlpc__learning_rate':['constant', 'invscaling', 'adaptive']}

parameter_grid = {
    'mlpc__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'mlpc__activation': ['relu', 'tanh', 'logistic'],
    'mlpc__solver': ['adam', 'sgd', 'lbfgs'],
    #'mlpc__alpha': [0.0001, 0.001, 0.01],
    'mlpc__learning_rate': ['constant', 'adaptive', 'invscaling'],
    #'mlpc__learning_rate_init': [0.001, 0.01, 0.1],
    'mlpc__max_iter': [200, 300, 400],
    #'mlpc__tol': [1e-4, 1e-3, 1e-2],
    #'mlpc__batch_size': [32, 64, 128],
    #'mlpc__momentum': [0.9, 0.95, 0.99],
    #'mlpc__early_stopping': [True, False],
    #'mlpc__validation_fraction': [0.1, 0.2, 0.3],
    #'mlpc__beta_1': [0.9, 0.95, 0.99],
    #'mlpc__beta_2': [0.999, 0.995, 0.99]
}


gs = GridSearchCV(pl, parameter_grid,  cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)