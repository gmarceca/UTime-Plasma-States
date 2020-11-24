'''
A randomly sampling batch sequence object
Performs class-balanced sampling across uniformly randomly selected PlasmaShot
objects.
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
from utime.sequences import UTimeRandomDataFetcherEndtoEndWOffset
from utime.errors import NotLoadedError

class UTimeRandomDataGenerator(UTimeRandomDataFetcherEndtoEndWOffset):
    '''
    UTimeRandomDataFetcher sub-class that samples class-balanced random batches
    across (uniformly) randomly selected PlasmaShot objects with calls to
    self.__getitem__.

    The 'sample_prob' property can be set to a list of values in [0...1] with
    sum == 1.0 that gives the probability that a plasma state of
    ground truth label matching the sample_prob index will be sampled.

    ''' 
    def __init__ (self, identifier, dataset, batch_size, data_per_period, cfg_dic, n_channels, margin=0, logger=None , no_log=False, debug=False):
        
        self.batch_size = batch_size
        
        self.offset = 0
        self.lstm_time_spread = cfg_dic['lstm_time_spread']
        self.no_input_channels = n_channels
        self.no_classes = len(cfg_dic['states'])
        self.data_per_period = data_per_period
        self.batch_shape = [batch_size, self.lstm_time_spread//self.data_per_period, self.data_per_period, self.no_input_channels]
        
        if self.no_classes == 3: 
            self.sample_prob = [1.0/3, 1.0/3, 1.0/3] # prob for [L, D, H] sampling
            self.states = {0:'L', 1:'D', 2:'H'}
        elif self.no_classes == 2:
            self.sample_prob = [1.0/2, 1.0/2] # prob for [L, H] sampling
            self.states = {0:'L', 1:'H'}

        self.cfg_dic = cfg_dic
        
        self.debug = debug

        super().__init__(identifier=identifier,
                         dataset=dataset,
                         batch_size=batch_size,
                         data_per_period=data_per_period,
                         cfg_dic=cfg_dic,
                         n_channels=n_channels,
                         margin=margin)

    def log(self):
        """ Log basic information on this object """
        self.logger("[*] UTimeDataGenerator initialized: {}\n"
                    "    Batch shape:     {}\n"
                    "    Sample prob.:    {}\n"
                    "    N pairs:         {}\n"
                    "    Margin:          {}\n"
                    "    data_per_period:          {}\n"
                    "    no_input_channels:          {}\n"
                    "    no_classes:      {}\n").format(self.identifier, 
                                                self.batch_shape,
                                                self.sample_prob,
                                                len(self.dataset),
                                                margin,
                                                self.data_per_period,
                                                self.no_input_channels,
                                                self.no_classes)


    def __getitem__(self, index):
        ''' 
        debug = True 
        is intended to be used just to instance a UTimeDataGenerator
        object and plot some distributions, but not to train a model
        '''
        
        self.seed()
        return self.get_class_balanced_random_batch()

    
    @property
    def shape(self):
        return self.batch_shape     
            
    #def on_epoch_end(self):
    #    print('\n Generator epoch finished.')
    #    for generator in self.sub_generators:
    #        generator.on_epoch_end()

    def get_class_balanced_random_period(self):
        # Get random class according to the sample probs.
        classes = np.arange(self.no_classes)
        cls = self.states[np.random.choice(classes, size=1, p=self.sample_prob)[0]]
        found = False
        while not found:
            data = np.random.choice(self.dataset)
            if not data.loaded:
                raise NotLoadedError
            assert len(data) > 0
            if cls not in data.class_to_period_dict or \
                len(data.class_to_period_dict[cls]) == 0:
                # This PS does not have the given class
                continue
            try:
                # Get the period index of a randomly sampled class (according
                # to sample_prob distribution) within the PlasmaShot pair
                
                idx = np.random.choice(data.class_to_period_dict[cls], 1)[0]
                s_ind = idx + self.margin + self.offset
                e_ind = idx + self.margin + self.offset + self.lstm_time_spread//self.data_per_period
                
                min_id = self.offset
                if s_ind < min_id:
                    temp = min_id - s_ind
                    s_ind += temp
                    e_ind += temp

                max_id = data.plasma_states.shape[0] - self.offset -1

                if e_ind > max_id:
                    temp = e_ind - max_id
                    e_ind -= temp
                    s_ind -= temp
                assert s_ind >= min_id
                assert e_ind <= max_id

                selected_indices = np.arange(s_ind, e_ind)
                
                X_, y_ = self.data_generation(data, selected_indices)
                #X_ = np.ones((100, 30, 1))
                #y_ = np.ones(100)
                return X_, y_
            except KeyError:
                continue


    def get_class_balanced_random_batch(self):

        X, y = [], []
        while len(X) != self.batch_size:
            X_, y_ = self.get_class_balanced_random_period()
            X.append(X_), y.append(y_)
        return self.process_batch(X, y)
