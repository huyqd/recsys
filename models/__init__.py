from collections import OrderedDict
from .baseline import Popularity
from .matrix_factorization import AlsMF

MODELS = OrderedDict([
    ("Popularity", Popularity),
    ("AlsMF", AlsMF),
])
