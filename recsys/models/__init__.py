from .baseline import *
from .mf import *
from .nn import *
from .ae import *
from collections import OrderedDict

BINARY_MODELS_DICT = OrderedDict([
    ("Popularity", Popularity),
    ("AlsMF", AlsMF),
    ("VanillaMF", VanillaMF),
    ("BiasMF", BiasMF),
    ("GMF", GMF),
    ("MLP", MLP),
    ("NeuMF", NeuMF),
])

RATING_MODELS_DICT = OrderedDict([
    ("AutoEncoder", AutoEncoder),
])
