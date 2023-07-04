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


def get_ndcg(true, pred):
    match = pred.eq(true).nonzero(as_tuple=True)[1]
    ncdg = torch.log(torch.Tensor([2])).div(torch.log(match + 2))
    ncdg = ncdg.sum().div(pred.shape[0]).item()

    return ncdg


def get_apak(true, pred):
    k = pred.shape[1]
    apak = pred.eq(true).div(torch.arange(k) + 1)
    apak = apak.sum().div(pred.shape[0]).item()

    return apak


def get_hr(true, pred):
    hr = pred.eq(true).sum().div(pred.shape[0]).item()

    return hr


def get_eval_metrics(scores, true, k=10):
    test_items = [torch.LongTensor(list(item_scores.keys())) for item_scores in scores]
    test_scores = [torch.Tensor(list(item_scores.values())) for item_scores in scores]
    topk_indices = [s.topk(k).indices for s in test_scores]
    topk_items = [item[idx] for item, idx in zip(test_items, topk_indices)]
    pred = torch.vstack(topk_items)
    ndcg = get_ndcg(true, pred)
    apak = get_apak(true, pred)
    hr = get_hr(true, pred)

    return ndcg, apak, hr
