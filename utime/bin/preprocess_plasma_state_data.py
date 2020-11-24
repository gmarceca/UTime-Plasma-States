import numpy as np
import pandas as pd
from utime.bin.window_functions import *
from utime.bin.label_smoothing import *
from utime.bin.helper_funcs import *
import pickle
import time
from collections import defaultdict
import argparse
import os
from glob import glob

def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = argparse.ArgumentParser(description="Preprocess data for dataset preparation")
    parser.add_argument("--data_dir", type=str,
                        help="Path to output data directory")
    parser.add_argument("--machine", type=str,
                        help="Specify machine")
    return parser

class IDsAndLabels(object):
    def __init__(self,):
        self.ids = {}
        self.len = 0
    
    def __len__(self,):
        return self.len
    
    def generate_id_code(self, shot, index):
        return str(str(shot)+'/'+str(index))
    
    def add_id(self, shot, k, transitions, elms, dithers):
        code = self.generate_id_code(shot, k)
        if code in self.ids.keys():
            return
        else:
            self.ids[code] = {'transitions': transitions, 'elms':elms, 'dithers': dithers}
            self.len += 1
            # print('added id', self.len, len(self.ids))
    
    def get_sorted_ids(self):
        return sorted(self.ids.keys(), key = lambda ind: int(self.get_shot_and_id(ind)[1]))
    
    def get_ids(self):
        # print 'getting ids', self.len, len(self.ids), len(self.ids.keys())
        return list(self.ids.keys())
    
    def get_shot_and_id(self, ID):
        s_i = ID.split('/')
        return s_i[0], int(s_i[1])
            
    def get_label(self, ID):
        return self.ids[ID]
    
    def get_ids_and_labels(self):
        # pairs = np.empty((self.len, 2))
        pairs = []
        sorted_ids = sorted(self.get_ids(), key = lambda ind: int(self.get_shot_and_id(ind)[1]))
        for ind in sorted_ids:
            pairs += [[ind, self.get_label(ind)]]
        return pairs

    def get_shots(self):
        shots = []
        for ID in self.get_sorted_ids():
            shot, ind = self.get_shot_and_id(ID)
            shots += [str(shot)]
        return set(shots)
    
class IDsAndLabelsLSTM(IDsAndLabels):
    def __init__(self,):
        IDsAndLabels.__init__(self)
    
    def add_id(self, shot, k, elm_lab_in_dt, state_lab_in_dt): #state
        code = self.generate_id_code(shot, k)
        self.ids[code] = (elm_lab_in_dt, state_lab_in_dt)
        self.len += 1

def first_prepro_cycle(shots_and_labelers, shots_and_labelers_dic, data_dir, gaussian_hinterval, machine):
    
    intc_times = {}
    shot_dfs = {}
    
    for s in shots_and_labelers:
        
        labelers = shots_and_labelers_dic[s.split('-')[0]]

        intc_times[s.split('-')[0]] = load_shot_and_equalize_times(data_dir, s.split('-')[0], labelers, machine)
        fshot, fshot_times = load_fshot_from_labeler(s, data_dir, machine)
        #fshot['sm_elm_label'], fshot['sm_non_elm_label'] = smoothen_elm_values(fshot.ELM_label.values, smooth_window_hsize=gaussian_hinterval)
        #fshot['sm_none_label'], fshot['sm_low_label'], fshot['sm_high_label'], fshot['sm_dither_label'] = smoothen_states_values_gauss(fshot.LHD_label.values, fshot.time.values, smooth_window_hsize=gaussian_hinterval)
        
        #CUTOFF to put all elm labels at 0 where state is not high
        #fshot.loc[fshot['LHD_label'] != 3, 'ELM_label'] = 0
        
        #fshot = state_to_trans_event_disc(fshot, gaussian_hinterval)
        #fshot = trans_disc_to_cont(fshot, gaussian_hinterval)
        
        shot_dfs[str(s)] = fshot.copy()

    return shot_dfs, intc_times

def second_prepro_cycle(shots_and_labelers, shot_dfs, itsc_times):
    
    print('Second prepro cycle')

    for shot in shots_and_labelers:
        shot_no = shot[:5]
        labeler_intersect_times = itsc_times[shot_no]
        fshot = shot_dfs[str(shot)].copy()
        fshot = fshot[fshot['time'].round(5).isin(labeler_intersect_times)]
        fshot = normalize_signals_mean(fshot) #NORMALIZATION CAN ONLY HAPPEN AFTER SHOT FROM BOTH LABELERS HAS BEEN ASSERTED TO BE THE SAME!
        shot_dfs[str(shot)] = fshot

    return shot_dfs

def third_prepro_cycle(shots_and_labelers, shot_dfs, data_dir_out):
    
    print('Third prepro cycle')

    ids_low = IDsAndLabelsLSTM()
    ids_dither = IDsAndLabelsLSTM()
    ids_high = IDsAndLabelsLSTM()

    ids = {'L': ids_low, 'D': ids_dither, 'H': ids_high}
    
    for shot in shots_and_labelers:
        l_set = []
        l_time = []
        l_index_start = 0
        l_index_start_tmp = 0
        l_index_end = 0

        d_set = []
        d_time = []
        d_index_start = 0
        d_index_start_tmp = 0
        d_index_end = 0

        h_set = []
        h_time = []
        h_index_start = 0
        h_index_start_tmp = 0
        h_index_end = 0

        fshot = shot_dfs[str(shot)]
        for k in range(len(fshot)):
            dt = fshot.iloc[k]
            elm_lab_in_dt = get_elm_label_in_dt(dt)
            state_lab_in_dt = get_state_labels_in_dt(dt)
            if (int(dt.LHD_label) == 1):
                if (k == (l_index_start_tmp + 1)): # If it's a consecutive index, i.e belong to same region
                    l_index_start_tmp = k
                    l_index_end = k
                else:
                    if (l_index_start < l_index_end):
                        l_set.append((shot + '/' + str(l_index_start), shot + '/' + str(l_index_end)))
                        l_time.append((shot + '/' + str(fshot['time'].values[l_index_start]), shot + '/' + str(fshot['time'].values[l_index_end])))
                    l_index_start = k
                    l_index_start_tmp = l_index_start
                ids['L'].add_id(shot, k, elm_lab_in_dt, state_lab_in_dt) 
            
            elif (int(dt.LHD_label) == 2):
                
                if (k == (d_index_start_tmp + 1)): # If it's a consecutive index, i.e belong to same region
                    d_index_start_tmp = k
                    d_index_end = k
                else:
                    if (d_index_start < d_index_end):
                        d_set.append((shot + '/' + str(d_index_start), shot + '/' + str(d_index_end)))
                        d_time.append((shot + '/' + str(fshot['time'].values[d_index_start]), shot + '/' + str(fshot['time'].values[d_index_end])))
                    d_index_start = k
                    d_index_start_tmp = d_index_start
                ids['D'].add_id(shot, k, elm_lab_in_dt, state_lab_in_dt)
            
            elif (int(dt.LHD_label) == 3):
                
                if (k == (h_index_start_tmp + 1)): # If it's a consecutive index, i.e belong to same region
                    h_index_start_tmp = k
                    h_index_end = k
                else:
                    if (h_index_start < h_index_end):
                        h_set.append((shot + '/' + str(h_index_start), shot + '/' + str(h_index_end)))
                        h_time.append((shot + '/' + str(fshot['time'].values[h_index_start]), shot + '/' + str(fshot['time'].values[h_index_end])))
                    h_index_start = k
                    h_index_start_tmp = h_index_start
                ids['H'].add_id(shot, k, elm_lab_in_dt, state_lab_in_dt)
        
        if (l_index_start < l_index_end):
            l_set.append((shot + '/' + str(l_index_start), shot + '/' + str(l_index_end))) # Add the final tuple
            l_time.append((shot + '/' + str(fshot['time'].values[l_index_start]), shot + '/' + str(fshot['time'].values[l_index_end])))
        if (d_index_start < d_index_end):
            d_set.append((shot + '/' + str(d_index_start), shot + '/' + str(d_index_end))) # Add the final tuple
            d_time.append((shot + '/' + str(fshot['time'].values[d_index_start]), shot + '/' + str(fshot['time'].values[d_index_end])))
        if (h_index_start < h_index_end):
            h_set.append((shot + '/' + str(h_index_start), shot + '/' + str(h_index_end))) # Add the final tuple
            h_time.append((shot + '/' + str(fshot['time'].values[h_index_start]), shot + '/' + str(fshot['time'].values[h_index_end])))
        if l_set:
            np.save(os.path.join(data_dir_out, 'list_IDs_{}_{}.npy'.format(shot,'L')), l_set)
            np.save(os.path.join(data_dir_out, 'list_times_{}_{}.npy'.format(shot,'L')), l_time)
        if d_set:
            np.save(os.path.join(data_dir_out, 'list_IDs_{}_{}.npy'.format(shot,'D')), d_set)
            np.save(os.path.join(data_dir_out, 'list_times_{}_{}.npy'.format(shot,'D')), d_time)
        if h_set:
            np.save(os.path.join(data_dir_out, 'list_IDs_{}_{}.npy'.format(shot,'H')), h_set)
            np.save(os.path.join(data_dir_out, 'list_times_{}_{}.npy'.format(shot,'H')), h_time)

    return ids


def PreprocessingOffline(data_dir_in, data_dir_out, shot_ids, labelers, gaussian_hinterval, previous_shots, machine):
    ''' Preprocess data offline. Store labels in a .json dictionary and
    preprocess data in .npy files. This is intended to use it just one
    and hence avoid performing same preprocessing each time for running
    an experiment.'''
    
    shots_and_labelers = []
    shots_and_labelers_dic = defaultdict(list)

    for s in shot_ids:
        for l in labelers:
            if ((l == 'ffelici' or l == 'maurizio' or l == 'labit') and s not in previous_shots):
                print('Skipping shot {}'.format(str(s) + '-' + str(l)))
                continue
            # Skip preprocessing if it was already done
            if os.path.exists(os.path.join(data_dir_out,'shot_{}.pkl'.format(str(s) + '-' + str(l)))):
                continue
            if os.path.exists(data_dir_in + l + '/' + machine + '_'  + str(s).split('-')[0] + '_' + l + '_labeled.csv'):
                shots_and_labelers += (str(s) + '-' + str(l),)
                shots_and_labelers_dic[str(s)].append(l)
    
    if not shots_and_labelers:
        return

    shot_dfs, itsc_times = first_prepro_cycle(shots_and_labelers, shots_and_labelers_dic, data_dir_in, gaussian_hinterval, machine)
    
    if not shot_dfs:
        return

    shot_dfs = second_prepro_cycle(shots_and_labelers, shot_dfs, itsc_times)
    #ids_labels = third_prepro_cycle(shots_and_labelers, shot_dfs, data_dir_out)
    
    # Save pre-process data
    for i, sdf in enumerate(shot_dfs.values()):
        
        with open(os.path.join(data_dir_out,'shot_{}.pkl'.format(list(shot_dfs.keys())[i])), 'wb') as f:
            pickle.dump(sdf, f)
    
def run(args):
    
    machine = args.machine
 
    data_dir_in = os.path.abspath(args.data_dir)
    data_dir_in = os.path.join(data_dir_in, 'Validated/')

    data_dir_out = os.path.abspath(args.data_dir)
    data_dir = os.path.abspath(args.data_dir)
    start = time.time()
    labelers = ['ApauMarceca']

    # TCV shots
    #previous_shots = [30268, 61057, 30290, 30197, 43454, 30310, 60097, 32794, 60275, 33942, 33281, 42514, 62744, 30225, 29511, 34010, 31211, 34309, 32911, 31807, 33459, 57218, 32191, 58460, 31554, 30262, 45105, 53601, 47962, 61021, 31839, 33638, 31650, 31718, 45103, 32592, 30044, 33567, 26383, 52302, 32195, 26386, 59825, 33271, 56662, 57751, 58182, 33188, 30043, 32716, 42197, 33446, 48580, 57103]
    previous_shots = []

    gaussian_time_window = 10e-4
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate) # Deprecated

    # Get Clusters Hierarchy structure and preprocess shots
    # Load a matlab .mat file as np arrays
    from scipy.io import loadmat
    if machine == 'TCV':
        DWT_out = loadmat(os.path.join(data_dir, 'Clusters_DWT_26052020.mat'))
    elif machine == 'JET':
        DWT_out = loadmat(os.path.join(data_dir, 'Clusters_DTW_JET.mat'))
    # Get shotlist and clusters obtained from DTW tool
    shotlist = DWT_out['shotlist']
    if machine == 'TCV':
        clusters = DWT_out['myclusters']
    elif machine == 'JET':
        clusters = DWT_out['clusters']
    
    #FIXME add by hand the shot 34010 which is not in the Clusters...
    #shotlist = [np.array([[34010]])]
    #clusters = np.array([[1]])
    #print('clusters: ', clusters)
    #print('shotlist: ', shotlist)
    
    # Flat arrays
    shotlist = [item[0][0].flat[0] for item in shotlist]
    clusters = [item[0].flat[0] for item in clusters]

    all_shots = [int(f_.split('/')[-1].split('_')[1]) for lab in labelers for f_ in glob(os.path.join(data_dir_in, lab + '/' + machine + '*'))]
    print(all_shots)

    # Check all shots in dir are in the Cluster structure
    for s in all_shots:
        if s not in shotlist:
            #raise ValueError('shot {} not in cluster structure'.format(s))
            print('shot {} not in cluster structure. Assigning them to a specific cluster'.format(s))
            residual_cluster = np.max(clusters) + 1
            clusters.append(residual_cluster)
            shotlist.append(s)

    n_clusters = len(set(clusters))
    
    for i in np.arange(1, n_clusters+1):
        
        print('Start preprocessing of cluster %i' % i)

        # Get shots associated with cluster i
        shots = np.array(shotlist)[np.where(np.array(clusters) == i)[0]]
        data_dir_tmp = os.path.join(data_dir_out, 'cluster_{}'.format(str(i)))
        
        if not os.path.exists(data_dir_tmp):
            print("Creating directory at %s" % data_dir_tmp)
            os.makedirs(data_dir_tmp)

        PreprocessingOffline(data_dir_in, data_dir_tmp, shots, labelers, gaussian_hinterval, previous_shots, machine)
    
    end = time.time()
    print('elapsed time for data preprocessing: ', end - start)


def entry_func(args=None):
    parser = get_argparser()
    run(parser.parse_args(args))

if __name__ == "__main__":
    entry_func()
