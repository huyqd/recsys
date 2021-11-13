from collections import OrderedDict
from .baseline import Popularity
from .matrix_factorization import AlsMF, TorchMF
from .nn import *

MODELS = OrderedDict([
    ("Popularity", Popularity),
    ("AlsMF", AlsMF),
    ("TorchMF", TorchMF),
    ("GMF", GMF),
    ("MLP", MLP),
    ("NeuMF", NeuMF),
])
