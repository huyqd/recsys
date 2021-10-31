import torch


def get_ncdg(true, pred):
    match = pred.eq(true).nonzero(as_tuple=True)[1]
    ncdg = torch.log(torch.Tensor([2])).div(torch.log(match + 2))
    ncdg = ncdg.sum().div(pred.shape[0]).mul(100).item()

    return ncdg


def get_apak(true, pred):
    k = pred.shape[1]
    apak = pred.eq(true).div(torch.arange(k) + 1)
    apak = apak.sum().div(pred.shape[0]).mul(100).item()

    return apak


def get_hr(true, pred):
    hr = pred.eq(true).sum().div(pred.shape[0]).mul(100).item()

    return hr


def get_eval_metrics(scores, ds, k=10):
    pred = scores.topk(k, dim=1).indices
    true = ds.test_pos[:, [1]]
    ncdg = get_ncdg(true, pred)
    apak = get_apak(true, pred)
    hr = get_hr(true, pred)

    return ncdg, apak, hr
