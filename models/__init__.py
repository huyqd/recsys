from collections import OrderedDict
from .baseline import Popularity
from .matrix_factorization import AlsMF, TorchMF

MODELS = OrderedDict([
    ("Popularity", Popularity),
    ("AlsMF", AlsMF),
    ("TorchMF", TorchMF),
])
