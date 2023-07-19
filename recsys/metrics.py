import torch
import numpy as np


def ndcg_score(true, pred):
    """Compute ndcg score given true values and predicted values"""
    # Discount array
    discount = np.log2(np.arange(2, pred.shape[1] + 2)).reshape(1, -1)

    # Get relevance, i.e. check if any items in pred matches any items in true
    rel = np.array([np.isin(pred[i], true[i]) for i in range(true.shape[0])])

    # Get ideal relevance
    irel = np.zeros(pred.shape)
    irel[:, : true.shape[1]] = 1

    # Compute dcg, idcg
    dcg = np.divide(rel, discount).sum(axis=1)
    idcg = np.divide(irel, discount).sum(axis=1)

    return (dcg / idcg).mean()


def hr_score(true, pred):
    """Compute hr score given true values and predicted values"""
    # Get relevance, i.e. check if any items in pred matches any items in true
    rel = np.array([np.isin(true[i], pred[i]) for i in range(true.shape[0])])

    return rel.mean()
