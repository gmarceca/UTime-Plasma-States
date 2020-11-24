"""
Script that prepares a folder of data for cross-validation experiments by
randomly splitting the dataset into partitions and storing links to the
relevant files in sub-folders for each split.
"""

from glob import glob
import os
import numpy as np
import random
from MultiPlanarUNet.utils import create_folders
import argparse
from collections import defaultdict

# These values are normally overwritten from the command-line, see argparser
_DEFAULT_TEST_SHOTS = 0
_DEFAULT_VAL_FRACTION = 0.20


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = argparse.ArgumentParser(description="Prepare a data folder for a"
                                                 "CV experiment setup.")
    parser.add_argument("--data_dir", type=str,
                        help="Path to data directory")
    parser.add_argument("--machine", type=str,
                        help="Path to data directory")
    parser.add_argument("--subject_dir_pattern", type=str,
                        help="Glob-like pattern used to select subject folders")
    parser.add_argument("--CV", type=int, default=5,
                        help="Number of splits (default=5)")
    parser.add_argument("--bootstrap_split", action="store_true",
                        help="Botstrap criteria to create train/val split")
    parser.add_argument("--selected_test_set", action="store_true",
                        help="Pass test set selected a priori")
    parser.add_argument("--out_dir", type=str, default="",
                        help="Directory to store CV subfolders "
                             "(default=views")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files to CV-subfolders instead of "
                             "symlinking (not recommended)")
    parser.add_argument("--file_list", action="store_true",
                        help="Create text files with paths pointing to the "
                             "files needed under each split instead of "
                             "symlink/copying. This is usefull on systems "
                             "were symlink is not supported, but the dataset "
                             "size is too large to store in copies. "
                             "NOTE: Only one of --copy and --file_list "
                             "flags must be set.")
    parser.add_argument("--validation_fraction", type=float,
                        default=_DEFAULT_VAL_FRACTION,
                        help="Fraction of OVERALL data size used for "
                             "validation in each split. In a 5-CV setting with"
                             " N=100 and val_frac=0.20, each split will have "
                             "N_train=60, N_val=20 and N_test=20 "
                             "subjects/records")
    parser.add_argument("--test_shots", type=int,
                        default=_DEFAULT_TEST_SHOTS,
                        help="Fraction of data size used for test if CV=1.")
    parser.add_argument("--shots_per_cluster", type=int,
                        default=1,
                        help="Number of shots per cluster requested for val / test split")
    parser.add_argument("--common_prefix_length", type=int, required=False,
                        help="If specified, files of identical naming in the"
                             " first 'common_prefix_length' letters will be"
                             " considered a single entry. This is useful for"
                             " splitting multiple studies on the same subject"
                             " as together.")
    return parser


def assert_dir_structure(data_dir, out_dir):
    """ Asserts that the data_dir exists and the out_dir does not """
    if not os.path.exists(data_dir):
        raise OSError("Invalid data directory '%s'. Does not exist." % data_dir)
    if os.path.exists(out_dir):
        raise OSError("Output directory at '%s' already exists." % out_dir)

def assert_cluster_size (clusters, N):
    """ Asserts that the number of clusters is higher than the ones requested """
    if not (len(clusters) > N):
        raise ValueError("Number of clusters {} must be higher than the ones requested {}".format(len(clusters), N))

def assert_shots_donot_overlap (train, val, test):
    """ Asserts that the train/val/test shots do not overlap """
    
    for s in train:
        if s in val or s in test:
            raise ValueError("training shot {} must not be in validation or test sets".format(s))

    for s in val:
        if s in test:
            raise ValueError("val shot {} must not be in test set".format(s))

def create_view_folders(out_dir, n_splits):
    """
    Helper function that creates a set of 'split_0', 'split_1'..., folders
    within a directory 'out_dir'. If n_splits == 1, only creates the out_dir.
    """
    if not os.path.exists(out_dir):
        print("Creating directory at %s" % out_dir)
        os.makedirs(out_dir)
    for i in range(n_splits):
        split_dir = os.path.join(out_dir, "split_%i" % i)
        print("Creating directory at %s" % split_dir)
        os.mkdir(split_dir)


def add_files(in_path, file_paths, out_folder, link_func=os.symlink):
    """
    Add all files pointed to by paths in list 'file_paths' to folder
    'out_folder' using the linking/copy function 'link_func'.

    Args:
        file_paths: A list of file paths
        out_folder: A path to a directory that should store the linked files
        link_func:  A function to apply on relative file paths in 'file_paths'
                    and absolute paths in 'file_paths'.
                    Typically one of os.symlink, os.copy or
                    _add_to_file_list_fallback.
    """
    
    for cls in file_paths:
        for shots in file_paths[cls]:
            in_ = glob(os.path.join(in_path, 'cluster_' + str(cls), '*'+str(shots)+'*'))
            for file_ in in_:
                file_name = file_.split('/')[-1]
                link_func(file_, out_folder + "/%s" % file_name)


def _add_to_file_list_fallback(rel_file_path,
                               file_path,
                               fname="LIST_OF_FILES.txt"):

    """
    On some system symlinks are not supported, if --files_list flag is set,
    uses this function to add each absolute file path to a list at the final
    subfolder that is supposed to store symlinks or actual files (--copy)

    At run-time, these files must be loaded by reading the path from this
    file instead.

    Args:
        rel_file_path: (string) Relative path pointing to the file from the
                                current working directory.
        file_path:     (string) Absolute path to the file
        fname:         (string) Filename of the file to store the paths
    """
    # Get folder where list of files should be stored
    folder = os.path.split(file_path)[0]

    # Get absolute path to file
    # We change dir to get the correct abs path from the relative
    os.chdir(folder)
    abs_file_path = os.path.abspath(rel_file_path)

    # Get path to the list of files
    list_file_path = os.path.join(folder, fname)

    with open(list_file_path, "a") as out_f:
        out_f.write(abs_file_path + "\n")


def pair_by_names(files, common_prefix_length=None):
    """
    Takes a list of file names and returns a list of tuples of file names in
    the list that share 'common_prefix_length' of identical leading characters

    That is, a list of files ['FILE_1_1', 'FILE_1_2', 'FILE_2_1'] and
    common_prefix_length 6 will result in:

       [ ('FILE_1_1', 'FILE_1_2') , ('FILE_2_1',) ]

    Args:
        files:                (list) A list of filenames
        common_prefix_length: (int)  A number of leading characters to match

    Returns:
        A list of tuples of paired filenames
    """
    from collections import defaultdict
    if common_prefix_length is not None:
        names = [os.path.split(i)[-1][:common_prefix_length] for i in files]
    else:
        names = [os.path.splitext(os.path.split(i)[-1])[0] for i in files]
    inds = defaultdict(list)
    for i, item in enumerate(names):
        inds[item].append(i)
    pairs = inds.values()
    return [tuple(np.array(files)[i]) for i in pairs]


def get_split_sizes(subject_dirs, n_splits, args, desc):
    """
    Returns the number of samples to include in the training, validation and
    testing sub-sets in each split given the parsed arguments (see argparser)

    Also prints the results to screen.

    Args:
        subject_dirs: (list)   List of all subject dirs in the dataset
        n_splits:     (int)    Number of splits to perform
        args:         (tuple)  Arguments passed, see argparser.
        desc:         (string) A string describing whether each entity in
                               'subject_dirs' is a 'subject' or 'record'.

    Returns:
        3 ints, number of training-, validation- and testing samples for each
        split.
    """
    n_total = len(subject_dirs)
    if n_splits > 1:
        n_test = int(np.ceil(n_total / n_splits))
    else:
        n_test = int(np.ceil(n_total * args.test_fraction))
    n_val = int(np.ceil(n_total * args.validation_fraction))
    if n_val + n_test >= n_total:
        raise ValueError("Too large test/validation_fraction - "
                         "No training samples left!")
    n_train = n_total - n_test - n_val
    print("-----")
    print("Total {}:".ljust(40).format(desc), n_total)
    print("(Approx.) Train {} pr. split:".ljust(40).format(desc), n_train)
    print("Validation {} pr. split:".ljust(40).format(desc), n_val)
    print("(Approx.) Test {} pr. split:".ljust(40).format(desc), n_test)
    return n_train, n_val, n_test


def run_on_split(in_path, split_path, train_data, val_data, test_data, n_train, n_val, n_test, args):
    """
    Add the train/val/test files of a single split to the split directories

    Depending on the arguments parsed (--copy, --file_list etc.) either copies,
    symlinks or creates a LIST_OF_FILES.txt file of absolute paths in each
    split sub-directory.

    Args:
        in_path: (string) Path to original files
        split_path:      (string) Path to the split directory
        train_split:      (list)   List of paths pointing to split train data
        val_data:  (list)   List of paths pointing to the split val data
        test_data:  (list)   List of paths pointing to the split test data
        n_train:           (bool)    Copy train set split
        n_val:           (bool)    Copy validation set split
        n_test:           (bool)    Copy test set split
        args:            (tuple)  Parsed arguments, see argparser.
    """

    # Define train, val and test sub-dirs
    train_path = os.path.join(split_path, "train") if n_train else None
    val_path = os.path.join(split_path, "val") if n_val else None
    test_path = os.path.join(split_path, "test") if n_test else None

    # Create folders if not existing
    create_folders([train_path, val_path, test_path])

    # Copy or symlink?
    if args.copy:
        from shutil import copyfile
        move_func = copyfile
    elif args.file_list:
        move_func = _add_to_file_list_fallback
    else:
        move_func = os.symlink

    
    # Add training
    if n_train:
        add_files(in_path, train_data, train_path, move_func)
    # Add test data
    if n_test:
        add_files(in_path, test_data, test_path, move_func)
    if n_val:
        # Add validation
        add_files(in_path, val_data, val_path, move_func)


def run(args):
    """
    Run the script according to 'args' - Please refer to the argparser.
    """
    data_dir = os.path.abspath(args.data_dir)
    machine = args.machine
    n_shots = int(args.test_shots)
    selected_test_set = args.selected_test_set

    if selected_test_set and n_shots > 0:
        raise ValueError("Test set cannot be specific and random at the same time.")
    
    shots_per_cluster = int(args.shots_per_cluster)
    n_splits = int(args.CV)
    #if n_splits > 1:
    out_dir = os.path.join(data_dir, args.out_dir, "%i_CV" % n_splits)
    #else:
    #    out_dir = os.path.join(data_dir, args.out_dir, "fixed_split")

    #if n_splits == 1 and not args.test_shots:
    #    raise ValueError("Must specify --test_fraction with --CV=1.")
    if args.copy and args.file_list:
        raise ValueError("Only one of --copy and --file_list "
                         "flags must be set.")
    
    #FIXME
    n_val = True

    # Assert suitable folders
    assert_dir_structure(data_dir, out_dir)

    # Create sub-folders
    create_view_folders(os.path.join(data_dir, out_dir), n_splits)

    # Get subject dirs 'cluster_*'
    subject_dirs = glob(os.path.join(data_dir, args.subject_dir_pattern))
    
    # Get cluster --> shots mapping
    cluster_map_shots = defaultdict(set)
    for cl_dir in subject_dirs:
        
        # Assume cluster index is the last character of the string
        cl_index = int(cl_dir.split('/')[-1].split('_')[-1])
        # Get shots_ids for each cluster
        shots_ids = glob(os.path.join(cl_dir, 'shot_*'))
        for sh in shots_ids:
            # Get shot number from string and added to dic
            # Assume shot length is 5
            shot_id = sh.split('/')[-1].split('_')[-1][:5]
            cluster_map_shots[cl_index].add(shot_id)

    
    # ======================= Start Test split ============================

    # Get clusters which have more than one shot
    # cluster_map_shots = {1: {'34010', '...'}, 2: {'32195', '...'}, ... : ...}
    cluster_map_shots_reduced = {i[0]:i[1] for i in cluster_map_shots.items() if len(i[1]) >= shots_per_cluster}
    
    # Assert number of clusters > number requested
    assert_cluster_size(cluster_map_shots_reduced, n_shots)
    
    # Get list
    if selected_test_set:
        if machine == 'TCV':
            test_shots_list = ['59073', '61714', '61274', '59065', '61010', '61043', '64770', '64774', '64369', '64060',
                '64662', '64376', '57093', '57095', '61021', '32911', '30268', '45105', '62744', '60097', '58460', '61057', '31807', '33459', '34309', '53601', '42197']
        elif machine == 'JET':
            test_shots_list = ['94652', '94785', '94792', '94967', '94968', '94972', '94973', '95298', '97405', '97469', '97470', '97477', '97476', '97478', '97828', '94658']
        test_shots = defaultdict(set)

        for test_sh in test_shots_list:
            for cl in cluster_map_shots:
                if test_sh in cluster_map_shots[cl]:
                    test_shots[cl].add(test_sh)
    else:
        test_shots = defaultdict(set)
        test_random_indices_clusters = np.random.choice(list(cluster_map_shots_reduced.keys()), n_shots)
    
        for ran in test_random_indices_clusters:
            cl_shots = cluster_map_shots_reduced[ran]
            random_shot = np.random.choice(list(cl_shots), 1)[0]
            test_shots[ran].add(random_shot)
        test_shots_list = [item for sub in test_shots.values() for item in list(sub)]

    # Copy test files
    run_on_split(in_path=data_dir, 
            split_path=data_dir,
            train_data='',
            val_data='',
            test_data=test_shots,
            n_train=False,
            n_val=False,
            n_test=True,
            args=args)

    # Remove selected test shots from list
    for test_sh in test_shots_list:
        for cl in cluster_map_shots:
            if test_sh in cluster_map_shots[cl]:
                cluster_map_shots[cl].remove(test_sh)
    
    # ======================= Start CV split ============================
    if n_splits > 1:
        if args.bootstrap_split:
            # Apply CV bootstrap split
            print('Performing boostrap split')
            train_shots_lists_splits, val_shots_lists_splits = CV_bootstrap_split (cluster_map_shots, n_splits, shots_per_cluster, n_shots)
        else:
            # Apply CV k-fold split
            print('Performing k-fold CV split')
            train_shots_lists_splits, val_shots_lists_splits = CV_kfold_split (cluster_map_shots, n_splits)
    else:
        train_shots_lists_splits = [cluster_map_shots]
        val_shots_lists_splits = [cluster_map_shots]


    # Assert CV list split len matches n_splits
    if not (len(train_shots_lists_splits) == n_splits or len(val_shots_lists_splits) == n_splits):
        ValueError("Size of CV split list does not match n_split requested")
    
    # Perform the split according the train and shots selected
    split_counter = 0
    for train_shots, val_shots in zip(train_shots_lists_splits, val_shots_lists_splits):      
        
        train_shots_list = [item for sub in train_shots.values() for item in list(sub)]
        val_shots_list = [item for sub in val_shots.values() for item in list(sub)]

        split_path = os.path.join(data_dir, out_dir, 'split_{}'.format(split_counter))   
        
        if n_splits > 1:
            assert_shots_donot_overlap (train_shots_list, val_shots_list, test_shots_list)

        # Add/copy/symlink the files to the split directories
        run_on_split(in_path=data_dir, 
                split_path=split_path,
                train_data=train_shots,
                val_data=val_shots,
                test_data='',
                n_train=True,
                n_val=n_val,
                n_test=False,
                args=args)

        split_counter += 1
        pass

    pass


def CV_bootstrap_split (clusters, n_splits, shots_per_cluster, n_shots):
    """ Computes CV split via random sampling (bootstrap) 

    Inputs:
    
    clusters: dictionary that maps each cluster to a shot number
    
    n_splits: CV number of splits, default = 5
    
    shots_per_cluster: for the val set get the clusters which have >= shots in. This is to guarantee
    the same cluster is present in both val and train sets (in case shots_per_cluster > 1)
    
    n_shots: Number of shots required for the val_set (same as the test set).
    Caveat: the number of shots in each val/test sets might not be necessary equal to n_shots.
    In case that more than one labeler validated a shot you will end up with a higher number
    of files, i.e: shot_31554-labit.pkl, shot_31554-ffelici.pkl, etc. But each shot number
    is present only in one of the val or test sets but not in both.

    returns: list of len equal n_splits where each element is a dictionary {clusters:shots} for train and val sets
    """

    # Re-compute mapping
    cluster_map_shots_reduced = {i[0]:i[1] for i in clusters.items() if len(i[1]) >= shots_per_cluster}
    assert_cluster_size(cluster_map_shots_reduced, n_shots)
    
    train_shots_splits = []
    val_shots_splits = []
    
    # Get CV folds
    for s_ in np.arange(0, n_splits):
        
        # Get val list
        val_shots = defaultdict(set)
        random_indices_clusters = np.random.choice(list(cluster_map_shots_reduced.keys()), n_shots)

        for ran in random_indices_clusters:
            cl_shots = cluster_map_shots_reduced[ran]
            random_shot = np.random.choice(list(cl_shots), 1)[0]
            val_shots[ran].add(random_shot)
        val_shots_list = [item for sub in val_shots.values() for item in list(sub)]
        
        # Get training list
        train_shots = defaultdict(set)
        for cl in clusters:
            for sh in clusters[cl]:
                if sh not in val_shots_list:
                    train_shots[cl].add(sh)
        
        train_shots_splits.append(train_shots)
        val_shots_splits.append(val_shots)
        
    return train_shots_splits, val_shots_splits


def CV_kfold_split (clusters, n_splits):
    """ Computes CV split via k-fold split """
    
    # Get all shots from clusters dic
    all_shots = [sh for shs in clusters.values() for sh in shs]
    
    # Get shots -> cluster mapping (invert clusters dictionary)
    shots_map_cluster = {sh: cl for cl, shs in clusters.items() for sh in shs}

    # In-place shuffling
    np.random.shuffle(all_shots)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits)
    
    train_shots_splits = []
    val_shots_splits = []
    
    for train_index, test_index in kf.split(all_shots):

        # Get train and val shots fold
        train_shots, test_shots = np.asarray(all_shots)[train_index], np.asarray(all_shots)[test_index]

        # Initialize train/val cluster --> shots dics
        val_cluster_map_shots = defaultdict(set)
        train_cluster_map_shots = defaultdict(set)
        
        # Loop in train_shots selected from k-fold split
        for tr_sh in train_shots:
            # Get cluster tr_sh belongs to
            cl = shots_map_cluster[tr_sh]
            # Fill train cluster --> shot dic
            train_cluster_map_shots[cl].add(tr_sh)

        # Loop in val_shots selected from k-fold split
        for val_sh in test_shots:
            # Get cluster val_sh belongs to
            cl = shots_map_cluster[val_sh]
            # Fill val cluster --> shot dic
            val_cluster_map_shots[cl].add(val_sh)
        
        # Fill list that will be returned. Each element contains a fold train/val split dic
        train_shots_splits.append(train_cluster_map_shots)
        val_shots_splits.append(val_cluster_map_shots)

    return train_shots_splits, val_shots_splits

def entry_func(args=None):
    parser = get_argparser()
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
