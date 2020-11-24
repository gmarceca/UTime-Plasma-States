import pandas as pd
import numpy as np
from decimal import *
getcontext().prec = 3
import math
from collections import OrderedDict
import scipy
import datetime
from scipy import stats
import pickle
import copy
import sys


def batch_wrapper(generator, batch_size):
    X, y = [], []
    for batch in generator:
        X.append(batch[0]), y.append(batch[1])
        if len(X) == batch_size:
            yield X, y
            X, y = [], []
    if len(X) != 0:
        yield X, y
