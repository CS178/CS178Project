# -*- coding: utf-8 -*-
"""
Label Propagation
    Utilizes Sklearn's semi_supervised package to implement a KNN kernel
    based Label Propagration learner.
    DESCRIPTION
        The :mod:`sklearn.semi_supervised` module implements semi-supervised learning
        algorithms. These algorithms utilized small amounts of labeled data and large
        amounts of unlabeled data for classification tasks. This module includes Label
        Propagation. 
        
    kernel: KNN
        rbf kernel requires a complete graph and overruns in memory.
        Feature selector is required, and next step in project
    gamma: 20
        Parameter for rbf kernel
    max_iter: 30
        Complexity control for knn
    n_neighbors: 7
        Parameter for knn, how many neighbors to consider
    alpha: float
        Clamping factor    
    tol: 0.001
        Converenge tolerance: threshold to consider system at steady state
"""
from sklearn.semi_supervised import LabelPropagation
from sklearn import metrics
import numpy as np
 
#K nearest neighbors model ensures we dont run over our memory
#rbf Kernel needs complete graph, so requires feature selection
lp_model = LabelPropagation(kernel = 'knn') #Label Propagation model
Xtr = np.genfromtxt("data/Kaggle.X1.train.txt", delimiter = ',') #Get X training data
Ytr_labels = np.genfromtxt("data/Kaggle.Y.labels.train.txt",delimter = ','); #Get classification data


#Unlabeled points - random size for now. 
unlabeled_points = np.where(np.random.random_integers(0,1,size = len(Ytr_labels)))
labels = np.copy(Ytr_labels) #Save training labels for testing
labels[unlabeled_points] = -1  #Set unlabeled value, classes : 0, 1
lp_model.fit(Xtr,labels) #Train

#############################################
#   Models use n_neighbors and max_iteration to control kernel
#############################################

#############################################
#   Test Functions
#############################################
#Mean squared Error
yhat = lp_model.predict(Xtr);
mse = metrics.mean_squared_error(Ytr_labels,yhat);

###############################################
# Cross Validation
###############################################