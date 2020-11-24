import os
import numpy as np

from utime.dataset import PlasmaShot
from utime.errors import CouldNotLoadError
from utime.dataset.utils import find_subject_folders
from utime import defaults
from MultiPlanarUNet.logging import ScreenLogger
from glob import glob

class PlasmaShotDataset(object):
    """
    Represents a collection of PlasmaShot objects
    """
    
    def __init__(self, cfg_dic, debug, data_dir, period_length_sec=None, identifier=None, logger=False, no_log=False):
        
        self.cfg_dic = cfg_dic
        self.subject_dir = os.path.abspath(data_dir)
        
        # Read unprocessed csv files (data processing is done in the plasma_shot class)
        if self.cfg_dic['read_csv']:
            if self.cfg_dic['validate_score']:
                self.shots_files = [glob(os.path.join(self.subject_dir, self.cfg_dic['Machine'] + "_" + str(self.cfg_dic['shot']) + "_*_labeled.csv"))[0]]
            else: # This option is used for non validated shots (when ground-truth is not available yet)
                self.shots_files = [os.path.join(self.subject_dir, self.cfg_dic['Machine'] + "_" + str(self.cfg_dic['shot']) + "_signals.csv")]
        else:
            self.shots_files = glob(os.path.join(self.subject_dir, "shot_*.pkl"))
        
        self.period_length_sec = period_length_sec
        self.debug = debug

        self.all_shots = []
        self.pairs = []

        self.logger = logger or ScreenLogger()

        for sh in self.shots_files:
            # Get shot number from string and added to list all_shots
            if self.cfg_dic['read_csv']:
                sh_id = sh.split('/')[-1].split('_')[1]
            else:
                sh_id = sh.split('/')[-1].split('_')[-1]
            self.all_shots.append(sh_id)

        # Initialize PlasmaShot objects
        for shot in self.all_shots:
            if self.debug:
                ps = PlasmaShot(shot, self.subject_dir, self.cfg_dic, load=True, debug=self.debug)
            else:
                ps = PlasmaShot(shot, self.subject_dir, self.cfg_dic)
            self.pairs.append(ps)
        if len(np.unique([p.identifier for p in self.pairs])) != len(self.pairs):
            raise RuntimeError("Two or more PlasmaShot objects share the same"
                               " identifier, but all must be unique.")
        self._identifier = identifier or os.path.split(self.data_dir)[-1]
        if not no_log:
            self.log()

    def log(self, message=None):
        """ Log basic properties about this dataset """
        id_msg = "[Dataset: {}]".format(self.identifier)
        if message is None:
            message = str(self)
        self.logger("{} {}".format(id_msg, message))

    @property
    def identifier(self):
        """ Returns the dataset ID string """
        return self._identifier

    @property
    def loaded_pairs(self):
        """ Returns stored PlasmaShot objects that have data loaded """
        return [s for s in self if s.loaded]

    @property
    def non_loaded_pairs(self):
        """ Returns stored PlasmaShot objects that do not have data loaded """
        return [s for s in self if not s.loaded]

    def __len__(self):
        """ Returns the number of stored PlasmaShot objects """
        return len(self.pairs)

    def __getitem__(self, item):
        """ Return an element from the list of stored PlasmaShot objects """
        return self.pairs[item]

    def __iter__(self):
        """ Yield elements from the list of stored PlasmaShot objects """
        for pair in self.pairs:
            yield pair

    def load(self, N=None, random_order=True):
        """
        Load all or a subset of stored PlasmaShot objects
        Data is loaded using a thread pool with one thread per PlasmaShot.

        Args:
            N:              Number of PlasmaShot objects to load. Defaults to
                            loading all.
            random_order:   Randomly select which of the stored objects to load
                            rather than starting from the beginning. Only has
                            an effect with N != None
        Returns:
            self, reference to the PlasmaShotDataset object
        """
        from concurrent.futures import ThreadPoolExecutor
        if N is None:
            N = len(self)
            random_order = False
        not_loaded = self.non_loaded_pairs
        if random_order:
            to_load = np.random.choice(not_loaded, size=N, replace=False)
        else:
            to_load = not_loaded[:N]
        self.log("Loading {}/{} PlasmaShot objects...".format(len(to_load),
                                                              len(self)))
        pool = ThreadPoolExecutor(max_workers=min(len(to_load), 7))
        res = pool.map(lambda x: x.load(), to_load)
        try:
            for i, ss in enumerate(res):
                print(" -- {}/{}".format(i+1, len(to_load)), end="\r", flush=True)
        except CouldNotLoadError as e:
            raise CouldNotLoadError("Could not load plasma shot {}."
                                    " Please refer to the above "
                                    "traceback.".format(e.study_id)) from e
        finally:
            pool.shutdown()
        return self


    def get_batch_sequence(self,
                           batch_size,
                           random_batches=True,
                           balanced_sampling=True,
                           margin=0,
                           scaler=None,
                           batch_wise_scaling=False,
                           no_log=False,
                           **kwargs):
        """
        Return a utime.sequences BatchSequence object made from this dataset.
        A BatchSequence (sub derived) object is used to extract batches of data
        from all or individual PlasmaShot objects represented by this
        PlasmaShotDataset.

        All args pass to the BatchSequence object.
        Please refer to its documentation.

        Returns:
            A BatchSequence object
        """
        loaded = self.loaded_pairs
        if len(loaded) == 0:
            raise IndexError("At least 1 PlasmaShot pair must be loaded"
                             " at batch sequence creation time.")
        # Assure all same dpe
        dpe = np.asarray([l.data_per_period for l in loaded])
        if not np.all(dpe == dpe[0]):
            raise ValueError("'get_batch_sequence' currently requires all "
                             "PlasmaShot pairs to have an identical number of"
                             "data points pr. period. "
                             "Got: {}".format(dpe))
        # Assure all same channels
        cnls = np.asarray([l.n_sample_channels for l in loaded])
        if not np.all(cnls == cnls[0]):
            raise ValueError("'get_batch_sequence' currently requires all "
                             "PlasmaShot pairs to have an identical number of"
                             "channels. Got: {}".format(cnls))

        # Init and return the proper BatchSequence sub-class
        from utime.sequences import get_sequence_class_ps
        sequence_class = get_sequence_class_ps(random_batches, balanced_sampling)
        
        return sequence_class(identifier=self.identifier,
                              dataset=self.pairs,
                              batch_size=batch_size,
                              data_per_period=dpe[0],
                              n_channels=cnls[0],
                              cfg_dic=self.cfg_dic,
                              margin=margin,
                              logger=self.logger,
                              no_log=no_log
                              )
