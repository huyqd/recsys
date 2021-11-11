import torch


def get_ncdg(true, pred):
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
    ncdg = get_ncdg(true, pred)
    apak = get_apak(true, pred)
    hr = get_hr(true, pred)

    return ncdg, apak, hr
