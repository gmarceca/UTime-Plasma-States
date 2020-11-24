'''
The UTimeDataFetcher class implements the process_batch function
to properly reshape the whole input sequence for the evaluation step. 
'''

import numpy as np
import pandas as pd
import abc
import math
import sys
import random
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LogNorm
import glob
import pickle
from collections import defaultdict
from tensorflow.keras.utils import Sequence
from multiprocessing import current_process

from MultiPlanarUNet.logging import ScreenLogger

class UTimeDataFetcherEndtoEndWOffset(Sequence):

    def __init__ (self, identifier, dataset, batch_size, data_per_period, cfg_dic, n_channels, margin=0):

        self.is_seeded = {}
        
        self.lstm_time_spread = cfg_dic['lstm_time_spread']
        self.batch_size = batch_size
        self.dataset = dataset
        
        self.id_to_pair = {pair.identifier: pair for pair in self.dataset}

        self.n_channels = n_channels
        self.data_per_period = data_per_period
        self.logger = ScreenLogger()
        self.total_length = 0
       
        self.margin = margin
        self.identifier = identifier

        self.batch_scaler = None


    def seed(self):
        """
        If multiprocessing, the processes will inherit the RNG state of the
        main process - here we reseed each process once so that the batches
        are randomly generated across multi-processes calls to the Sequence
        batch generator methods

        If multi-threading this method will just re-seed the 'MainProcess'
        process once
        """
        pname = current_process().name
        if pname not in self.is_seeded or not self.is_seeded[pname]:
            # Re-seed this process
            np.random.seed()
            self.is_seeded[pname] = True

    def process_batch(self, X, y, copy=True):
        """
        Process a batch (X, y) of sampled data.

        The process_batch method should always be called in the end of any
        method that implements batch sampling.

        Processing includes:
          1) Casting of X to ndarray of dtype float32
          2) Ensures X has a channel dimension, even if self.n_channels == 1
          3) Ensures y has dtype uint8 and shape [-1, 1]
          4) Ensures both X and y has a 'batch dimension', even if batch_size
             is 1.
          5) If a 'batch_scaler' is set, scales the X data

        Args:
            X:     A list of ndarrays corresponding to a batch of X data
            y:     A list of ndarrays corresponding to a batch of y labels
            copy:  If True, force a copy of the X and y data. NOTE: data may be
                   copied in some cases even if copy=False, see np.asarray

        Returns:
            Batch of (X, y) data
            OBS: Currently does not return the w (weights) array
        """
        # Cast and reshape arrays
        arr_f = np.asarray if copy is False else np.array
        X = arr_f(X, dtype=np.float32).squeeze()
        if self.n_channels == 1:
            X = np.expand_dims(X, -1)
        if self.data_per_period == 1:
            X = np.expand_dims(X, -2)
        y = np.expand_dims(arr_f(y, dtype=np.uint8).squeeze(), -1)

        expected_dim = len(self.batch_shape)
        if X.ndim == expected_dim-1:
            X, y = np.expand_dims(X, 0), np.expand_dims(y, 0)
        elif X.ndim != expected_dim:
            raise RuntimeError("Dimensionality of X is {} (shape {}), but "
                               "expected {}".format(X.ndim, X.shape,
                                                    expected_dim))

        if self.batch_scaler:
            # Scale the batch
            self.scale(X)
        # w = np.ones(len(X))
        #print('process_batch y: ', y)

        return X, y  # , w  <- weights currently disabled, fix dice-loss first

