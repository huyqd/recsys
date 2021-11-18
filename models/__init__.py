from .baseline import *
from .mf import *
from .nn import *
from collections import OrderedDict

MODELS_DICT = OrderedDict([
    ("Popularity", Popularity),
    ("AlsMF", AlsMF),
    ("VanillaMF", VanillaMF),
    ("BiasMF", BiasMF),
    ("GMF", GMF),
    ("MLP", MLP),
    ("NeuMF", NeuMF),
])
