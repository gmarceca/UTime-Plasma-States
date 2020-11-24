"""
Implements the PlasmaShot class which represents a plasma discharge
"""
import os
import numpy as np
from contextlib import contextmanager
import pickle
from collections import defaultdict
from utime import defaults
from MultiPlanarUNet.logging import ScreenLogger
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from utime.bin.helper_funcs import *
from glob import glob

class PlasmaShot(object):
    """
    Represents a plasma discharge object
    """

    def __init__(self, shot_id, 
            subject_dir,
            cfg_dic,
            states=['L', 'D', 'H'],
            load=False,
            debug=False):
        
        '''
        shot_id: 'ShotNumber-labeler', i.e: 3211-ffelici
        subject_dir: Path to stored input files
        cfg_dic: cfg file dataset dictionary
        states: plasma states to load (for the labels only)
        '''
        self.debug = debug
        self.subject_dir = subject_dir
        self.states = cfg_dic['states']
        self.shot_id = shot_id

        if len(self.states) == 3:
            self.states_dic = {'L':0, 'D': 1, 'H': 2}
        elif len(self.states) == 2:
            self.states_dic = {'L':0, 'H': 1}
        
        self.time_spread = cfg_dic['lstm_time_spread']
        self.time_spread = self.time_spread + 10
        self.pad_seq = cfg_dic['pad_seq']
        self.points_per_window = cfg_dic['points_per_window']
        self.diagnostics = cfg_dic['diagnostics']
        self.add_FFT = cfg_dic['add_FFT']
        self.add_full_FFT = cfg_dic['add_full_FFT']
        self.plot_fft = cfg_dic['plot_FFT']
        self.project_dir = cfg_dic['project_dir']
        self.read_csv = cfg_dic['read_csv']
        self.validate_score = cfg_dic['validate_score']
        self.machine = cfg_dic['Machine']

        self.delta_fft = None
        self.sliding_step = None
        self.Np = None
        self.freqs = None
        self.Fs = None

        if self.add_FFT or self.add_full_FFT:
            self.delta_fft = cfg_dic['delta_fft']
            self.sliding_step = cfg_dic['sliding_step']
            self.Fs = cfg_dic['Fs']
            self.Np = self.power_bit_length(self.delta_fft)
            self.freqs = np.arange(0, self.Fs/2, self.Fs/self.Np)

        self.plasma_shot_df = None
        self.list_IDs = defaultdict(list)
        self._plasma_shot = None
        self._states = None
        self.crop = None
        
        if load:
            self.load()
        
        if self.debug:
            self.plot_period()

    @property
    def plasma_shot(self):
        """ Returns the PS object (an ndarray of shape [-1, n_channels]) """
        return self._plasma_shot

    @property
    def identifier(self):
        """
        Returns shot ID: "shot-labeler"
        """
        return self.shot_id[:-4] if not self.read_csv else self.shot_id
    
    @property
    def class_to_period_dict(self):
        return {c: np.where(self.plasma_states == self.states_dic[c])[0] for c in self.states}

    def __len__(self):
        """ Returns the size of the PlasmaShot = periods * data_per_period """
        return self._plasma_shot.shape[1] * self.points_per_window

    @property
    def plasma_states(self):
        """ Returns the plasma state labels """
        return self._states
    
    @property
    def plasma_times(self):
        return self._times

    def unload(self):
        """ Unloads the PlasmaShot and states data """
        self._plasma_shot = None
        self._states = None

    def reload(self, warning=True):
        """ Unloads and loads """
        if warning and self.loaded:
            print("Reloading PlasmaShot '{}'".format(self.identifier))
        self.unload()
        self.load()


    def get_all_periods(self):
        """
        Returns the full (dense) data of the PlasmaShot

        Returns:
            X: An ndarray of shape [self.n_periods,
                                    self.data_per_period,
                                    self.n_channels]
            y: An ndarray of shape [self.n_periods, 1]
        """
        X = self._plasma_shot.reshape(-1, self.data_per_period, self.n_channels)
        y = self.plasma_states
        if len(X) != len(y):
            err_msg = ("Length of PlasmaShot array does not match length dense "
                       "states array ({} != {}) ".format(len(X),len(y)))
            self.raise_err(ValueError, err_msg)
        return X, y

    @contextmanager
    def loaded_in_context(self):
        """ Context manager from automatic loading and unloading """
        self.load()
        try:
            yield self
        finally:
            self.unload()

    @property
    def loaded(self):
        """ Returns whether the PlasmaShot data is currently loaded or not """
        return not any((self.plasma_shot is None,
                        self.plasma_states is None))

    @property
    def data_per_period(self):
        """
        Computes and returns the data (samples) per period of
        'period_length_sec' seconds of time (en 'epoch' in sleep research)
        """
        return self.points_per_window

    @property
    def n_channels(self):
        """ Returns the number of channels in the PSG array """
        if self.add_FFT:
            return len(self.diagnostics) + 2
        elif self.add_full_FFT:
            return len(self.diagnostics) + self.freqs.shape[0]
        else:
            return len(self.diagnostics)

    @property
    def n_sample_channels(self):
        """
        Returns the number of channels that will be returned by
        self.extract_from_psg (this may be different from self.n_channels if
        self.channel_sampling_groups is set).
        """
        #if self.channel_sampling_groups:
        #    return len(self.channel_sampling_groups())
        #else:
        return self.n_channels

    def load_shot_pkl (self):

        with open(os.path.join(self.subject_dir, 'shot_{}'.format(self.shot_id)), 'rb') as f:
            self.plasma_shot_df = pickle.load(f)

    def load_shot_csv (self):
        if self.validate_score:
            self.plasma_shot_df = pd.read_csv(glob(os.path.join(self.subject_dir, self.machine + '_'  + self.shot_id + '_*_labeled.csv'))[0], encoding='utf-8')
        else:
            self.plasma_shot_df = pd.read_csv(os.path.join(self.subject_dir, self.machine + '_'  + self.shot_id + '_signals.csv'), encoding='utf-8')
        self.ori_plasma_shot_df = self.plasma_shot_df
    
    def preprocess_shot (self):
        self.plasma_shot_df = remove_current_30kA(self.plasma_shot_df)
        if self.validate_score:
            self.plasma_shot_df = remove_no_state(self.plasma_shot_df) # This is only possible when shot is Validated
        self.plasma_shot_df = remove_disruption_points(self.plasma_shot_df)
        self.plasma_shot_df = self.plasma_shot_df.reset_index(drop=True)
        self.intersect_times = np.round(self.plasma_shot_df.time.values,5)
        self.plasma_shot_df = self.plasma_shot_df[self.plasma_shot_df['time'].round(5).isin(self.intersect_times)]
        self.plasma_shot_df = normalize_signals_mean(self.plasma_shot_df)

    # Compute STD FFT
    def std_fft(self, PD, delta_fft, sliding_step, Np):
        
        std_ = np.zeros(PD.shape)
        for i in range(delta_fft+1, PD.shape[0], sliding_step):
            PD_mean = np.mean(PD[i-delta_fft:i]) # remove DC component
            shot_fft = np.fft.fft(PD[i-delta_fft:i]-PD_mean,Np)/delta_fft
            shot_fft = 2*np.abs(shot_fft[:int(Np/2)])
            std_[i] = np.std(shot_fft)
        return std_
    
    # Compute FFT weighted by inv freq
    def weight_sum_inv_freq_fft(self, PD, delta_fft, sliding_step, Np, freqs):
        
        out_ = np.zeros(PD.shape)
        for i in range(delta_fft+1, PD.shape[0], sliding_step):

            PD_mean = np.mean(PD[i-delta_fft:i]) # remove DC component
            shot_fft = np.fft.fft(PD[i-delta_fft:i]-PD_mean,Np)/delta_fft
            shot_fft = 2*np.abs(shot_fft[1:int(Np/2)])

            # Compute weigthed sum
            tot_ampl = np.sum(shot_fft)
            w_sum = 0
            for j, amp in enumerate(shot_fft):
                w_sum += amp/freqs[j+1]
            w_sum /= tot_ampl
            out_[i] = w_sum

        return out_

    # Compute full FFT
    def fft(self, PD, delta_fft, sliding_step, Np):

        fft_ = np.zeros((PD.shape[0], int(Np/2)))
        for i in range(delta_fft+1, PD.shape[0], sliding_step):
            PD_mean = np.mean(PD[i-delta_fft:i]) # remove DC component
            shot_fft = np.fft.fft(PD[i-delta_fft:i]-PD_mean,Np)/delta_fft
            shot_fft = 2*np.abs(shot_fft[:int(Np/2)])
            fft_[i:] = shot_fft
        return fft_

    def power_bit_length(self, x):
        return 2**(x-1).bit_length()

    def reshape_ (self, df_signal, indexes, points_per_window):
        ret = np.asarray([df_signal]).reshape(int(len(indexes)/points_per_window), points_per_window)
        return ret

    def concatenate_channels_and_reshape (self):
        
        # Crop the signal in order to be an exact multiple of points_per_window
        if self.crop:
            self.plasma_shot_df = self.plasma_shot_df[:-self.crop]
            
            if self.read_csv:
                self.ori_plasma_shot_df = self.ori_plasma_shot_df[:-self.crop]
                self.intersect_times = self.intersect_times[:-self.crop]
        
        if (self.plasma_shot_df.shape[0] % self.points_per_window is not 0):
            ValueError("sequence length {} is not divisible by points_per_window {}".format(self.plasma_shot_df.shape[0], self.points_per_window))
        
        self._plasma_shot = np.empty((1, self.plasma_shot_df.shape[0]//self.points_per_window, self.points_per_window, self.n_channels))
        
        if self.pad_seq and self.plasma_shot_df.shape[0] < self.time_spread:
                self._plasma_shot_padded = np.empty((1, self.time_spread//self.points_per_window, self.points_per_window, self.n_channels))

        for i, d in enumerate(self.diagnostics):
            sig = self.plasma_shot_df[d].values
            
            self._plasma_shot[0,:,:,i] = self.reshape_ (sig, np.arange(0, self.plasma_shot_df.shape[0]), self.points_per_window)
            
            if self.pad_seq and self.plasma_shot_df.shape[0] < self.time_spread:
                    self._plasma_shot_padded[0,:,0,i] = np.pad(self._plasma_shot[0,:,0,i], (0, (self.time_spread-self.plasma_shot_df.shape[0])//self.points_per_window), 'constant')
        
        if self.pad_seq and self.plasma_shot_df.shape[0] < self.time_spread:
            self._plasma_shot = self._plasma_shot_padded

        # Add two more channels related to the FFT features of the PD signal
        if self.add_FFT:
            # STD feature
            stdfft = self.std_fft(self.plasma_shot_df['PD'], self.delta_fft, self.sliding_step, self.Np)
            self._plasma_shot[0,:,:,len(self.diagnostics)] = self.reshape_ (stdfft, np.arange(0, self.plasma_shot_df.shape[0]), self.points_per_window)
            # Inv Freq feature
            wsumfft = self.weight_sum_inv_freq_fft(self.plasma_shot_df['PD'], self.delta_fft, self.sliding_step, self.Np, self.freqs)
            self._plasma_shot[0,:,:,len(self.diagnostics)+1] = self.reshape_ (wsumfft, np.arange(0, self.plasma_shot_df.shape[0]), self.points_per_window)
        
        elif self.add_full_FFT:
            # Compute full FFT
            fft = self.fft(self.plasma_shot_df['PD'], self.delta_fft, self.sliding_step, self.Np)
            
            for j in range(0, fft.shape[1]):
                # Adding each FFT frequency as an additional channel
                self._plasma_shot[0,:,:,len(self.diagnostics) + j] = self.reshape_ (fft[:,j], np.arange(0, self.plasma_shot_df.shape[0]), self.points_per_window)
            
            if self.plot_fft:
                p_ = os.path.join(self.project_dir, 'plots')
                if not os.path.exists(p_):
                    os.mkdir(p_)
                fig, ax = plt.subplots(figsize=(12, 6))
                im = plt.imshow(fft.T,aspect='auto',interpolation="none", 
                                cmap = 'viridis',norm = LogNorm(vmin=.01,vmax=1))
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Real Time Spectogram')
                plt.gca().invert_yaxis()
                plt.savefig(os.path.join(p_, 'fft_shot_{}.png'.format(self.shot_id[:-4])))
                plt.close()

        self._times = self.plasma_shot_df.time.values
        
    def downsample_labels (self):    
        """
        """
        labels = self.plasma_shot_df.LHD_label.values.tolist()
        states = []
        for k in range(int(len(labels)/self.points_per_window)):
            window = labels[k*self.points_per_window : (k+1)*self.points_per_window]
            # determine the label of a window by getting the dominant class
            label = max(set(window), key = window.count)
            # L mode: label = 1, D mode: label = 2, H mode: label = 3
            # rest one so label is 0, 1 or 2
            label -= 1
            states.append(label)
        
        self._states = np.asarray([states]).swapaxes(0, 1)
        if self.pad_seq and self._states.shape[0] < self.time_spread:
            self._states_pad = np.pad(self._states[:,0], (0,self.time_spread//self.points_per_window - self._states.shape[0]), 'constant')
            self._states_pad = np.expand_dims(self._states_pad, 1)
            self._states = self._states_pad
    def _load(self):
        
        if self.read_csv:
            self.load_shot_csv()
            self.preprocess_shot()
        else:
            self.load_shot_pkl()
        
        # deprecated
        #self.load_labels_ID()
        
        if self.plasma_shot_df.shape[0]%self.points_per_window is not 0:
            self.crop = self.plasma_shot_df.shape[0]%self.points_per_window

        # preprocess inputs and store them (from df to np arrays)
        self.concatenate_channels_and_reshape()
        # Data were validated at full (10 KHz) sampling freq for the labels. Downsample it to be less prune to overfitting
        # and more compatible with realistic errors in the validation (prior uncertainty)
        if not self.read_csv:
            self.downsample_labels()
        else:
            # Just set it with zero values
            self._states = np.zeros(self._plasma_shot.shape[1])

    def plot_period (self):
        
        X = self.plasma_shot.squeeze()
        PD = X.reshape(X.shape[0]*X.shape[1])
        times = self.plasma_times
        pred = self.plasma_states.squeeze()

        len_shot = PD.shape[0]
        len_seq = 1000
        
        slices = np.arange(0, len_shot//len_seq + 1, 3)
        
        print('self.identifier: ', self.identifier)

        for s in range(0,len(slices)-1):
            times_list = []
            PD_list = []
            pred_list = []
        
            for i in range(slices[s], slices[s+1]):
                times_ = times[i*len_seq:(i+1)*len_seq]
                PD_ = PD[i*len_seq:(i+1)*len_seq]
                pred_ = pred[i*len_seq//10:(i+1)*len_seq//10]
                times_list.append(times_)
                PD_list.append(PD_)
                pred_list.append(pred_)
        
            self.plot_all_signals_all_trans_slices(times_list, PD_list, pred_list, self.identifier, s)
        
        #if (len_shot%len_seq):
        #    times_ = times[-(len_shot%len_seq):]
        #    PD_ = PD[-(len_shot%len_seq):]
        #    pred_ = pred[-(len_shot%len_seq)//10:]
        #    
        #    self.plot_all_signals_all_trans_slices([times_], [PD_], [pred_], self.identifier, 'end')

        #    #for k in range(X.shape[0]):
        #    #print("label: ", y[k,0,0])
        #    #plot_all_signals_all_trans(elms[k], y[k,:,0], X[k,:,0,:], ts[k], trans[k])

    def load(self):
        """
        High-level function invoked to load the PlasmaShot data
        """
        if not self.loaded:
            try:
                self._load()
            except Exception as e:
                raise errors.CouldNotLoadError("Unexpected load error for sleep "
                                               "study {}. Please refer to the "
                                               "above traceback.".format(self.identifier),
                                               study_id=self.identifier) from e
        return self


    def plot_all_signals_all_trans_slices(self, times_list, PD_list, pred_list, shot, slice_):
   
        from matplotlib import colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np

        font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 16}
        import matplotlib
        matplotlib.rc('font', **font)
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        cs = ['r','g','y',colors['teal'],'c','m',colors['black']]

        #fig, axs = plt.subplots(figsize=(19,5), nrows=3)
        fig, axs = plt.subplots(3,1,figsize=(19,12))
        #fig, axs = plt.subplots(3)

        leg = []

        for i in range(0,len(axs)):
            
            times = times_list[i]
            PD = PD_list[i]
            pred = pred_list[i]
            
            axs[i].plot(times,PD, label='PD')
            axs[i].grid()
            axs[i].set_ylabel('Signal values (norm.)')
            axs[i].set_xlabel('t(s)')

            colors_dic = {0:'yellow',1:'green',2:'blue'}
        
            for k in range(pred.shape[0]-1):
                #print(k)
                x_axis = (times[(k+1)*10] + times[k*10])/2
                axs[i].vlines(x=x_axis, ymin=0, ymax=np.max(PD), linestyle='--', color = colors_dic[pred[k]], linewidth=2, alpha=0.5)

            
        import matplotlib.patches as mpatches
        l_patch = mpatches.Patch(color='yellow', label='L mode')
        d_patch = mpatches.Patch(color='green', label='D mode')
        h_patch = mpatches.Patch(color='blue', label='H mode')

        fig.legend(handles=[l_patch, d_patch, h_patch], loc=3, prop={'size': 16}, ncol=1)

        fig.suptitle('UTime predictions for shot {}'.format(str(shot)))
        plt.tight_layout()
        plt.savefig('plots/predictions_slice_{}_shot_{}.png'.format(slice_, shot))
        plt.close()
