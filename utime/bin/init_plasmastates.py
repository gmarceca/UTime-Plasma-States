"""
Script for initializing new U-Time project directories
"""

from argparse import ArgumentParser
import os


def get_parser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Create a new project folder')

    # Define groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required named arguments')
    optional = parser.add_argument_group('optional named arguments')
    defaults = os.path.split(__file__)[0] + "/defaults"

    required.add_argument('--name', type=str, required=True,
                        help='the name of the project folder')
    optional.add_argument('--root', type=str, default=os.path.abspath("./"),
                          help='a path to the root folder in '
                               'which the project will be initialized '
                               '(default=./)')
    optional.add_argument("--model", type=str, default="utime",
                          help="Specify a model type parameter file. One of: "
                               "{} (default 'utime')".format(",".join(os.listdir(defaults))))
    optional.add_argument("--data_dir", type=str, default=None,
                          help="Optional specification of path to dir "
                               "storing data")
    optional.add_argument("--fold", type=str, default=None,
                          help="Specify the fold number interested to run")
    optional.add_argument("--CV", type=str, default=None,
                          help="Specify the number of folds for CV")
    return parser


def copy_yaml_and_set_data_dirs(in_path, out_path, fold, cv, data_dir=None, data_test_dir=None):
    """
    Creates a YAMLHParams object from a in_path (a hyperparameter .yaml file),
    inserts the 'data_dir' argument into data_dir fileds in the .yaml file
    (if present) and saves the hyperparameter file to out_path.

    Note: If data_dir is set, it is assumed that the folder contains data in
          sub-folders 'train', 'val' and 'test' (not required to exist).

    args:
        in_path:  (string) Path to a .yaml file storing the hyperparameters
        out_path: (string) Path to save the hyperparameters to
        data_dir: (string) Optional path to a directory storing data to use
                           for this project.
    """
    from utime.hyperparameters import YAMLHParams
    hparams = YAMLHParams(in_path, no_log=True, no_version_control=True)
    
    #dataset = 'dataset_config'
    # Set values in parameter file and save to new location
    data_ids = ("train", "val", "test")
    for dataset in data_ids:
        if (dataset == "test"):
            path = os.path.join(data_test_dir, dataset) if data_test_dir else "Null"
            dataset = dataset + "_data"
        else:
            path = os.path.join(data_dir, dataset) if data_dir else "Null"
            dataset = dataset + "_data"
            if hparams.get(dataset) and not hparams[dataset].get("data_dir"): # if file wants to modify is dataset_1.yaml
                if fold==0 and cv==0 and dataset == 'val_data': # this is the case for the full training (train+val / test)
                    path = os.path.join(data_test_dir, 'test')
                hparams.set_value(dataset, "data_dir", path, True, True)
                project_dir = '/'.join(out_path.split('/')[:-2])
                hparams.set_value('dataset_config', "project_dir", project_dir, True, True)
            else: # if file wants to modify is hparams_plasma_states.yaml
                if cv > 0:
                    hparams.set_value("datasets", "dataset_1", "dataset_configurations/dataset_1_fold{}.yaml".format(cv), overwrite=True)
                    
                    # Expected dirs to store logs for each fold must be created
                    try:
                        os.makedirs(out_path[:out_path.rfind('/')] + '/logs_fold{}'.format(cv))
                    except FileExistsError:
                        # Already accepted to overwrite
                        pass
                else:
                    hparams.set_value("datasets", "dataset_1", "dataset_configurations/dataset_1.yaml", overwrite=True)
    hparams.save_current(out_path)


def init_project_folder(default_folder, preset, out_folder, CV, fold, data_dir=None, data_test_dir=None):
    """
    Create and populate a new project folder with default hyperparameter files.

    Args:
        default_folder: (string) Path to the utime.bin.defaults folder
        preset:         (string) Name of the model/preset directory to use
        out_folder:     (string) Path to the project directory to create and
                                 populate
        data_dir:       (string) Optional path to a directory storing data to
                                 use for this project.
    """
    # Copy files and folders to project dir, set data_dirs if specified
    in_folder = os.path.join(default_folder, preset)
    for dir_path, dir_names, file_names in os.walk(in_folder):
        for dir_name in dir_names:
            p_ = os.path.join(out_folder, dir_name)
            if not os.path.exists(p_):
                os.mkdir(p_)
        for file_name in file_names:
            in_file_path = os.path.join(dir_path, file_name)
            sub_dir = dir_path.replace(in_folder, "").strip("/")
            out_file_path = os.path.join(out_folder, sub_dir, file_name)
            
            if not bool(fold):
                data_dir_ = os.path.join(data_dir, "%i_CV" % CV, "split_full") 
                copy_yaml_and_set_data_dirs(in_file_path, out_file_path, fold, 0, data_dir_, data_test_dir)
                
                for cv in range(1,CV+1):
                    data_dir_ = os.path.join(data_dir, "%i_CV" % CV, "split_%i" % (cv - 1))
                    if in_file_path.split('/')[-1] == "hparams.yaml":
                        continue
                    else:
                        out_file_path_ = out_file_path[:-5] + '_fold{}'.format(cv) + '.yaml'
                        copy_yaml_and_set_data_dirs(in_file_path, out_file_path_, fold, int(cv), data_dir_, data_test_dir)
            else:
                data_dir_ = os.path.join(data_dir, "%i_CV" % CV, "split_%i" % (fold - 1))
                copy_yaml_and_set_data_dirs(in_file_path, out_file_path, fold, 0, data_dir_, data_test_dir)

def run(args):
    """
    Run this script with the specified args. See argparser for details.
    """
    default_folder = os.path.split(os.path.abspath(__file__))[0] + "/defaults"
    if not os.path.exists(default_folder):
        raise OSError("Default path not found at %s" % default_folder)
    root_path = os.path.abspath(args.root)
    data_dir = args.data_dir
    
    fold = int(args.fold) if args.fold else 0 
    CV = int(args.CV)
    
    if fold > CV:
        raise ValueError("fold number cannot exceed total number of folds")

    if data_dir:
        data_dir = os.path.abspath(data_dir)
    
    data_test_dir = data_dir
    
    # Validate project path and create folder
    if not os.path.exists(root_path):
        raise OSError("root path '{}' does not exist.".format(args.root))
    else:
        out_folder = os.path.join(root_path, args.name)
        if os.path.exists(out_folder):
            response = input("Folder at '{}' already exists. Overwrite? "
                             "Only parameter files will be replaced. "
                             "(y/N) ".format(out_folder))
            if response.lower() != "y":
                raise OSError(
                    "Folder at '{}' already exists".format(out_folder))
        try:
            os.makedirs(out_folder)
        except FileExistsError:
            # Already accepted to overwrite
            pass
    init_project_folder(default_folder, args.model, out_folder, CV, fold, data_dir, data_test_dir)


def entry_func(args=None):
    # Parse arguments
    parser = get_parser()
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
