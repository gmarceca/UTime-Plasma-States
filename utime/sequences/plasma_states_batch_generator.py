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
from utime.sequences import UTimeDataFetcherEndtoEndWOffset
from utime.errors import NotLoadedError

class UTimeDataGenerator(UTimeDataFetcherEndtoEndWOffset):
    '''
    UTimeDataFetcher sub-class that gets the full plasma sequence
    discharge for the final evaluation step.
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
            self.states = {0:'L', 1:'D', 2:'H'}
        elif self.no_classes == 2:
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
                                                'NaN',
                                                len(self.dataset),
                                                margin,
                                                self.data_per_period,
                                                self.no_input_channels,
                                                self.no_classes)
    
    @property
    def shape(self):
        return self.batch_shape     

    def get_pair_by_id(self, study_id):
        """
        Return a PlasmaShot object by its identifier string

        Args:
            study_id: String identifier of a specific PlasmaShot

        Returns:
            A stored PlasmaShot object
        """
        return self.id_to_pair[study_id]

    def get_single_study_full_seq(self, study_id):
        """
        Return all periods/epochs/segments of data (X, y) of a PlasmaShot.
        Differs only from 'PlasmaShot.get_all_periods' in that the batch is
        processed and thus may be scaled.

        Args:
            study_id: A string identifier matching a single PlasmaShot object

        Returns:
            X: ndarray of plasma sequence data, shape [-1, data_per_period, n_channels]
            y: ndarray of labels, shape [-1, 1]
        """
        ss = self.get_pair_by_id(study_id)
        return self.process_batch(*ss.get_all_periods())
