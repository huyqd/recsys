from collections import OrderedDict
from .baseline import Popularity
from .matrix_factorization import AlsMF, TorchMF
from .matrix_factorization import LightningMF

MODELS = OrderedDict([
    ("Popularity", Popularity),
    ("AlsMF", AlsMF),
    ("TorchMF", TorchMF),
])

LIGHTNING_UTILS = OrderedDict([
    ("LightningMF", LightningMF),
])
