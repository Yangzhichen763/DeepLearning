# -*- coding: utf-8 -*-
"""
Created on April 7, 2020

@author: Shiping Wang
  Email: shipingwangphd@gmail.com
  Date: April 14, 2020.
"""

from sklearn import metrics
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

'''
   Clustering accuracy
'''
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


'''
 Evaluation metrics of clustering performance
      ACC: clustering accuracy
      NMI: normalized mutual information
      ARI: adjusted rand index
'''
def clusteringMetrics(trueLabel, predictiveLabel):
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, predictiveLabel)

    # Normalized mutual information
    NMI = metrics.normalized_mutual_info_score(trueLabel, predictiveLabel)

    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, predictiveLabel)

    return ACC, NMI, ARI