"""
Script which predicts on a set of data and evaluates the performance by
comparing to the ground truth labels.
"""

import os
from argparse import ArgumentParser
import readline
import numpy as np
from glob import glob
import pickle
import itertools

readline.parse_and_bind('tab: complete')


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Evaluate a U-Time model.')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to U-Time project folder')
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--machine", type=str, default="TCV",
                        help="Specify the machine (e.g TCV or JET) for which a model should be used")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold number")
    parser.add_argument("--channels", nargs='*', type=str, default=None,
                        help="A list of channels to use instead of those "
                             "specified in the parameter file.")
    parser.add_argument("--one_shot", action="store_true",
                        help="Segment each SleepStudy in one forward-pass "
                             "instead of using (GPU memory-efficient) sliding "
                             "window predictions.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous results at the output folder')
    parser.add_argument("--no_save", action="store_true",
                        help="Do not save prediction files")
    parser.add_argument("--fill_sql_table", action="store_true",
                        help="Save output in a sql db")
    parser.add_argument("--no_save_true", action="store_true",
                        help="Save the true hypnogram in addition to the "
                             "predicted hypnogram. Ignored with --no_save.")
    parser.add_argument("--no_eval", action="store_true",
                        help="Perform no evaluation of the prediction performance. "
                             "No label files loaded when this flag applies.")
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--data_split", type=str, default="val_data",
                        help="Which split of data of those stored in the "
                             "hparams file should the evaluation be performed "
                             "on.")
    parser.add_argument("--plot_hypnograms", action="store_true",
                        help="Add plots comparing the predicted versus true"
                             " hypnograms to folder [out_dir]/plots/hypnograms.")
    parser.add_argument("--plot_CMs", action="store_true",
                        help="Add plots showing per-sample confusion matrices."
                             " The plots will be stored in folder "
                             "[out_dir]/plots/CMs")
    parser.add_argument("--weights_file_name", type=str, required=False,
                        help="Specify the exact name of the weights file "
                             "(located in <project_dir>/model/) to use.")
    parser.add_argument("--run_from_GUI", action="store_true",
                        help="Run predictions from GUI")
    parser.add_argument("--downsampling_postproc", action="store_true",
                        help="Downsampling predictions at post-processing step")
    parser.add_argument("--validate_score", action="store_true",
                        help="Display kappa scores predictions from a model")
    parser.add_argument("--shot", type=str, default="",
                        help="Specify shot you want to validate")
    parser.add_argument("--wake_trim_min", type=int, required=False,
                        help="Only evaluate on within wake_trim_min of wake "
                             "before and after sleep, as determined by true "
                             "labels")
    return parser


def assert_args(args):
    """ Not yet implemented """
    return


def get_out_dir(out_dir, dataset):
    """ Returns a new, dataset-specific, out_dir under 'out_dir' """
    out_dir = os.path.abspath(out_dir)
    out_dir = os.path.join(out_dir, dataset)
    return out_dir


def prepare_output_dir(out_dir, overwrite):
    """
    Checks if the 'out_dir' exists, and if not, creates it
    Otherwise, an error is raised, unless overwrite=True, in which case nothing
    is done.
    """
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    elif not overwrite:
        files = os.listdir(out_dir)
        if files:
            raise OSError("out_dir {} is not empty and --overwrite=False. Folder"
                          " contains the following files: {}".format(out_dir,
                                                                     files))


def get_logger(out_dir, overwrite, name="evaluation_log"):
    """
    Returns a Logger object for the given out_dir.
    The logger will throw an OSError if the dir exists and overwrite=False, in
    which case the script will terminate with a print message.
    """
    from MultiPlanarUNet.logging import Logger
    try:
        logger = Logger(out_dir,
                        active_file=name,
                        overwrite_existing=overwrite,
                        no_sub_folder=True)
    except OSError:
        from sys import exit
        print("[*] A logging file 'logs/{}' already exists. "
              "If you wish to overwrite this logfile, set the --overwrite "
              "flag.".format(name))
        exit(0)
    return logger


def get_and_load_model(project_dir, hparams, logger, fold, machine, weights_file_name=None):
    """
    Initializes a model in project_dir according to hparams and loads weights
    in .h5 file at path 'weights_file_name' or automatically determined from
    the 'model' sub-folder under 'project_dir' if not specified.

    Args:
        project_dir:        Path to project folder
        hparams:            A YAMLHParams object storing hyperparameters
        logger:             A Logger object
        weights_file_name:  Optional path to .h5 parameter file

    Returns:
        Parameter-initialized model
    """
    if not weights_file_name:
        from utime.models.model_init import init_and_load_best_model
        
        if bool(fold):
            model_dir = os.path.join(project_dir, "model_fold{}".format(fold))
        else:
            if machine != "TCV":
                model_dir = os.path.join(project_dir, "model_{}".format(machine))
            else:
                model_dir = os.path.join(project_dir, "model")
        model, _ = init_and_load_best_model(
            hparams=hparams,
            model_dir=model_dir,
            logger=logger,
            by_name=True
        )
    else:
        from utime.models.model_init import init_and_load_model
        weights_file_name = os.path.join(project_dir, "model", weights_file_name)
        model = init_and_load_model(hparams=hparams,
                                    weights_file=weights_file_name,
                                    logger=logger,
                                    by_name=True)
    return model


def get_and_load_one_shot_model(full_state, project_dir,
                                hparams, logger, fold, machine, weights_file_name=None):
    """
    Returns a model according to 'hparams', potentially initialized from
    parameters in a .h5 file 'weights_file_name'.

    Independent of the settings in 'hparams', the returned model will be
    configured in 'one shot' mode - that is the model will predict on an entire
    shot input in one forward pass. The 'full_state' array is used to
    determine the corresponding number of segments.

    Args:
        full_state:     Array of sleep stage labels, shape [n_periods, 1]
        project_dir:        Path to project directory
        hparams:            YAMLHparams object
        logger:             Logger object
        weights_file_name:  Optional path to .h5 parameter file

    Returns:
        Initialized model
    """
    # Set seguence length
    n_periods = full_state.shape[0]
    hparams["build"]["batch_shape"][1] = n_periods
    hparams["build"]["batch_shape"][0] = 1  # Should not matter
    return get_and_load_model(project_dir, hparams, logger, fold, machine, weights_file_name)


def set_gpu_vis(num_GPUs, force_GPU, logger=None):
    """ Helper function that sets the GPU visibility as per parsed args """
    if force_GPU:
        from MultiPlanarUNet.utils.system import set_gpu
        set_gpu(force_GPU)
    else:
        # Automatically determine GPUs to use
        from MultiPlanarUNet.utils.system import GPUMonitor
        GPUMonitor(logger).await_and_set_free_GPU(num_GPUs, stop_after=True)


def plot_hypnogram(out_dir, pred, id_, true=None):
    """
    Wrapper around hypnogram plotting function
    """
    from utime.evaluation.plotting import plot_and_save_hypnogram
    hyp_plot_dir = os.path.join(out_dir, "plots", "hypnograms")
    plot_and_save_hypnogram(out_path=os.path.join(hyp_plot_dir, id_ + ".png"),
                            y_pred=pred,
                            y_true=true,
                            id_=id_)


def plot_cm(out_dir, pred, true, n_classes, id_):
    """
    Wrapper around confusion matrix plotting function
    """
    from utime.evaluation.plotting import plot_and_save_cm
    cm_plot_dir = os.path.join(out_dir, "plots", "CMs")

    # Compute and plot CM
    plot_and_save_cm(out_path=os.path.join(cm_plot_dir, id_ + ".png"),
                     pred=pred,
                     true=true,
                     n_classes=n_classes,
                     id_=id_,
                     normalized=True)


def save_sql(pred, y, shot_id, save_dir):

    import sqlite3
    connection = sqlite3.connect(os.path.join(save_dir, "files/predictions.db"))
    crsr = connection.cursor()
    
    sql_command = """CREATE TABLE IF NOT EXISTS emp (
    shot_id INTEGER NOT NULL,
    timestamp VARCHAR(10) NOT NULL,
    pred INTEGER,
    true INTEGER,
    PRIMARY KEY (shot_id, timestamp)
    );"""
    
    crsr.execute(sql_command)
    
    times = range(0, len(pred))
    
    # Loop through each time step and fill the database for a given shot_id
    for pred, true, t in zip(pred, y, times):
        try:
            crsr.execute('INSERT INTO emp (shot_id, timestamp, pred, true) VALUES (?,?,?,?);', [shot_id, str(t), str(pred), str(true[0])])
        except sqlite3.IntegrityError:
            crsr.execute('UPDATE emp SET (pred, true)=(?,?) WHERE shot_id==? AND timestamp==?;', [str(pred), str(true[0]), shot_id, str(t)])
    
    ## print sql table
    #sql_command = """SELECT * FROM emp;"""
    #crsr.execute(sql_command)
    #
    #print(crsr.fetchall())
    
    connection.commit()
    connection.close()    

def save(arr, fname):
    """
    Helper func to save an array (.npz) to disk in a potentially non-existing
    tree of sub-dirs
    """
    d, _ = os.path.split(fname)
    if not os.path.exists(d):
        os.makedirs(d)
    np.savez(fname, arr)


def _predict_sequence(study_pair, seq, model, verbose=True):
    """
    Predict on 'study_pair' wrapped by 'seq' using 'model'
    Predicts in batches of size seq.batch_size (set in hparams file)

    Args:
        study_pair: A SleepStudyPair object to predict on
        seq:        A BatchSequence object that stores 'study_pair'
        model:      An initialized and loaded model to predict with
        verbose:    Verbose level (True/False)

    Returns:
        An array of predicted sleep stages for all periods in 'study_pair'
        Shape [n_periods, n_classes]
    """
    from utime.utils.scriptutils.predict import sequence_predict_generator
    gen = seq.single_study_seq_generator(study_id=study_pair.identifier,
                                         overlapping=True)
    pred = sequence_predict_generator(model=model,
                                      total_seq_length=study_pair.n_periods,
                                      generator=gen,
                                      argmax=False,
                                      overlapping=True,
                                      verbose=verbose)
    return pred


def _predict_sequence_one_shot(study_pair, seq, model):
    """
    Predict on 'study_pair' wrapped by 'seq' using 'model'
    Assumes len(shot) (number of periods in a shot) is equal to the number of
    periods output by 'model' in a single pass (one-shot segmentation).

    Used with get_and_load_one_shot_model function (--one_shot set in args)

    Args:
        study_pair: A PlasmaStudyPair object to predict on
        seq:        A BatchSequence object that stores 'study_pair'
        model:      An initialized and loaded model to predict with

    Returns:
        An array of predicted plasma states for all periods in 'study_pair'
        Shape [n_periods, n_classes]
    """
    X, _ = seq.get_single_study_full_seq(study_pair.identifier)
    if X.ndim == 3:
        X = np.expand_dims(X, 0)
    return model.predict_on_batch(X)[0]

def predict_on(study_pair, seq, model=None, model_func=None, argmax=True):
    """
    High-level function for predicting on a single Shot
    object using a model returned when calling 'model_func'.

    Arguments 'model' and 'model_func' are exclusive, exactly one must be set

    Args:
        study_pair:  A Shot dataframe to predict on
        dataset_cfg: A dataset_config dictionary
        seq:         A dictionary that stores 'study_pair'
        model:       An initialized and loaded model to predict with
        model_func:  A callable which returns an intialized model
        argmax:      If true, returns [n_periods, 1] sleep stage labels,
                     otherwise returns [n_periods, n_classes] softmax scores.

    Returns:
        An array of predicted sleep stage scores for 'study_pair'.
        Shape [n_periods, 1] if argmax=True, otherwise [n_periods, n_classes]
    """
    if bool(model) == bool(model_func):
        raise RuntimeError("Must specify either model or model_func, "
                           "got both or neither.")
        pass
   
    y = study_pair.plasma_states

    # One-shot sequencing
    pred_func = _predict_sequence_one_shot
    # Get one-shot model of input shape matching the plasma sequence
    if model_func:
        model = model_func(y)
    # Get prediction
    pred = pred_func(study_pair, seq, model)
    if argmax:
        pred = np.argmax(pred, -1)
    return y, pred


def get_sequencer(dataset, hparams):
    """
    Returns a BatchSequence object (see utime.seqeunces)

    OBS: Initializes the BatchSequence with scale_assertion, and
    requires_all_loaded flags all set to False.

    args:
        dataset: (PlasmaShotDataset) A PlasmaShotDataset storing data to
                                     predict on
        hparams: (YAMLHparams)       Hyperparameters to use for the prediction

    Returns:
        A BatchSequence object
    """
    dataset.pairs[0].load()  # get_batch_sequence needs 1 loaded study
    hparams["fit"]["balanced_sampling"] = False
    seq = dataset.get_batch_sequence(random_batches=False,
                                     **hparams["fit"])
    return seq


def downsample_labels (states, pred, points_per_window):
    """
    """
    states = states.tolist()
    states_resampled = []
    pred_resampled = []
    for k in range(int(len(states)/points_per_window)):
        window = states[k*points_per_window : (k+1)*points_per_window]
        window_pred = pred[k*points_per_window : (k+1)*points_per_window]
        assert(len(window) == points_per_window)
        # determine the label of a window by getting the dominant class
        label = max(set(window), key = window.count)
        avg_pred = np.mean(window_pred, 0) # Mean probability in a given window for each state.
        # Since the DIS tool paints the graph based on the probability, we set the predictions to
        # 0 or 1 values, otherwise we end up with phases shorter than points_per_window in the GUI. FIXME
        #number_states = np.array([2])
        #avg_pred = np.zeros((number_states.size, number_states.max()+1))
        #avg_pred[np.arange(number_states.size),label] = 1
        #avg_pred = avg_pred[0]
        # L mode: label = 0, D mode: label = 1, H mode: label = 2
        states_resampled.append(label)
        pred_resampled.append(avg_pred)
    return np.asarray(states_resampled), np.asarray(pred_resampled)

def run_pred_and_eval(dataset,
                      out_dir,
                      model,
                      model_func,
                      hparams,
                      args,
                      shots_dir,
                      logger,
                      machine):
    """
    Run evaluation (predict + evaluate) on a all entries of a PlasmaShotDataset

    Args:
        dataset:     A dictionary {shot-id: dataframe}
        dataset_cfg: A dataset_config dictionary
        out_dir:     Path to directory that will store predictions and
                     evaluation results
        model:       An initialized model used for prediction
        model_func:  A callable that returns an initialized model for pred.
        hparams:     An YAMLHparams object storing all hyperparameters
        args:        Passed command-line arguments
        logger:      A Logger object
    """
    from MultiPlanarUNet.evaluate.metrics import dice_all, class_wise_kappa
    from utime.evaluation.dataframe import (get_eval_plasma_states_df, add_to_eval_df,
                                            log_eval_df, with_grand_mean_col)
    logger("\nPREDICTING ON {} STUDIES".format(len(dataset.pairs)))
    seq = get_sequencer(dataset, hparams)

    # Prepare evaluation data frames
    dice_eval_df = get_eval_plasma_states_df(seq)
    kappa_eval_df = get_eval_plasma_states_df(seq)

    # Predict on all samples
    for i, plasma_shot in enumerate(dataset):
        id_ = plasma_shot.identifier
        logger("[{}/{}] Predicting on PlasmaShot: {}".format(i+1,
                                                             len(dataset),
                                                             id_))
    
        # Predict
        with logger.disabled_in_context(), plasma_shot.loaded_in_context():

            if args.run_from_GUI:
                y, pred = predict_on(study_pair=plasma_shot,
                                 seq=seq,
                                 model=model,
                                 model_func=model_func, argmax=False)
                pred_max = np.argmax(pred, -1)
                
                if args.validate_score:
                    out = glob(os.path.join(out_dir, "files/{}".format(id_)+'-*'))
                    fn = os.path.join(out[0], "true.npz")
                    with np.load(fn) as Tdata:
                        y = Tdata['arr_0']

            else: # returns argmax(pred)
                y, pred = predict_on(study_pair=plasma_shot,
                             seq=seq,
                             model=model,
                             model_func=model_func)
                pred_max = pred
        
        if not args.no_save:
            # Save the output
            save_dir = os.path.join(out_dir, "files/{}".format(id_))
            save(pred_max, fname=os.path.join(save_dir, "pred.npz"))
            if not args.no_save_true:
                save(y, fname=os.path.join(save_dir, "true.npz"))

            if args.fill_sql_table:
                # Save the output in a sql db
                save_sql(pred_max, y, id_, out_dir)

        # Flag dependent evaluations: #TODO
        if args.plot_hypnograms:
            plot_hypnogram(out_dir, pred_max, id_, true=y)
        if args.plot_CMs:
            plot_cm(out_dir, pred_max, y, seq.no_classes, id_)

        # Evaluate: dice scores
        dice_pr_class = dice_all(y, pred_max,
                                 n_classes=seq.no_classes,
                                 ignore_zero=False, smooth=0)
        
        if args.validate_score or not args.run_from_GUI:
            logger("-- Dice scores:  {}".format(np.round(dice_pr_class, 4)))
        
        # Evaluate: kappa
        kappa_pr_class = class_wise_kappa(y, pred_max, n_classes=seq.no_classes,
                                          ignore_zero=False)
        if args.validate_score or not args.run_from_GUI:        
            logger("-- Kappa scores: {}".format(np.round(kappa_pr_class, 4)))
        
        if not args.run_from_GUI:
            add_to_eval_df(dice_eval_df, id_, values=dice_pr_class) 
            add_to_eval_df(kappa_eval_df, id_, values=kappa_pr_class)

    if not args.run_from_GUI:
        # Log eval to file and screen
        dice_eval_df = with_grand_mean_col(dice_eval_df)
        log_eval_df(dice_eval_df.T,
                    out_csv_file=os.path.join(out_dir, "evaluation_dice.csv"),
                    out_txt_file=os.path.join(out_dir, "evaluation_dice.txt"),
                    logger=logger, round=4, txt="EVALUATION DICE SCORES")
        kappa_eval_df = with_grand_mean_col(kappa_eval_df)
        log_eval_df(kappa_eval_df.T,
                    out_csv_file=os.path.join(out_dir, "evaluation_kappa.csv"),
                    out_txt_file=os.path.join(out_dir, "evaluation_kappa.txt"),
                    logger=logger, round=4, txt="EVALUATION KAPPA SCORES")
    else: 
        # Generate a dataframe with the appropiate format that the GUI needs. Check the following README for the format specifications:
        # https://gitlab.epfl.ch/spc/tcv/event-detection/UTime-PlasmaStates-V2/doc/csv_file_spec.md
        points_per_window = hparams["build"]["batch_shape"][2]
        upsampling_factor = points_per_window
        # Get probabilities and repeat x"points_per_window" the predictions to be same size as the original sampling frequency (10 KHz)
        
        decision_classes = np.argmax(pred, -1)
        # Downsampling decision_classes: compute max across labels in a window of size points_per_window
        if args.downsampling_postproc:
            downsampling_factor = 20
            # Downsample predictions by downsampling_factor
            decision_classes, pred = downsample_labels (decision_classes, pred, downsampling_factor)
            upsampling_factor = downsampling_factor
             
        LHD_det = repelem(decision_classes, upsampling_factor)
        ELM_det = repelem(np.zeros(pred.shape[0]), upsampling_factor)
        ELM_prob = repelem(np.zeros(pred.shape[0]), upsampling_factor)
        
        if machine == 'TCV':
           L_prob = repelem(pred[:,0], upsampling_factor)
           D_prob = repelem(pred[:,1], upsampling_factor)
           H_prob = repelem(pred[:,2], upsampling_factor)
        elif machine == 'JET':
           L_prob = repelem(pred[:,0], upsampling_factor)
           H_prob = repelem(pred[:,1], upsampling_factor)
           D_prob = repelem(np.zeros(pred.shape[0]), upsampling_factor)
        out_df = plasma_shot.ori_plasma_shot_df.copy()
        
        # Remove last remainder times from intersect_times to make it size compatible with predictions due to downsampling
        if args.downsampling_postproc:
            remainder = plasma_shot.intersect_times.shape[0]%LHD_det.shape[0]
            plasma_shot.intersect_times = plasma_shot.intersect_times[:-remainder]
        
        out_df['LHD_det'] = np.zeros(plasma_shot.ori_plasma_shot_df.shape[0])
        out_df.loc[out_df['time'].round(5).isin(plasma_shot.intersect_times), 'LHD_det'] = LHD_det
        
        out_df['ELM_det'] = np.zeros(plasma_shot.ori_plasma_shot_df.shape[0])
        out_df.loc[out_df['time'].round(5).isin(plasma_shot.intersect_times), 'ELM_det'] = ELM_det
        
        out_df['ELM_prob'] = np.zeros(plasma_shot.ori_plasma_shot_df.shape[0])
        out_df.loc[out_df['time'].round(5).isin(plasma_shot.intersect_times), 'ELM_prob'] = ELM_prob
        
        out_df['L_prob'] = np.zeros(plasma_shot.ori_plasma_shot_df.shape[0])
        out_df.loc[out_df['time'].round(5).isin(plasma_shot.intersect_times), 'L_prob'] = L_prob
        
        out_df['D_prob'] = np.zeros(plasma_shot.ori_plasma_shot_df.shape[0])
        out_df.loc[out_df['time'].round(5).isin(plasma_shot.intersect_times), 'D_prob'] = D_prob
        
        out_df['H_prob'] = np.zeros(plasma_shot.ori_plasma_shot_df.shape[0])
        out_df.loc[out_df['time'].round(5).isin(plasma_shot.intersect_times), 'H_prob'] = H_prob
        
        out_df.to_csv(columns=['time', 'IP', 'SXR', 'FIR', 'PD', 'DML', 'LHD_det', 'ELM_det', 'L_prob', 'D_prob', 'H_prob', 'ELM_prob'],
                          path_or_buf=os.path.join(shots_dir, machine + '_'  + str(args.shot) + '_UTime_det.csv'), index=False)
        
def repelem(arr, num):
    arr = list(itertools.chain.from_iterable(itertools.repeat(x, num) for x in arr.tolist()))
    return np.asarray(arr)

def cross_dataset_eval(dataset_eval_dirs, out_dir):
    """ Not implemented yet """
    pass


def run(args):
    """
    Run the script according to args - Please refer to the argparser.
    """
    assert_args(args)
    # Check project folder is valid
    from utime.utils.scriptutils.scriptutils_plasma_states import (assert_project_folder_ps,
                                         get_splits_from_all_datasets_ps)
    
    project_dir = os.path.abspath(args.project_dir)
    if args.run_from_GUI:
        project_dir = os.path.join(project_dir, 'algorithms/GMUTime/UTime-PlasmaStates/in_dir_eval')
    
    assert_project_folder_ps(project_dir, args.fold, evaluation=True)

    machine = args.machine
    # Prepare output dir
    if not args.run_from_GUI:
        out_dir = get_out_dir(args.out_dir, args.data_split)
        if bool(args.fold):
            out_dir = out_dir + "_fold{}".format(str(args.fold))
        prepare_output_dir(out_dir, args.overwrite)
        logger = get_logger(out_dir, args.overwrite)
    else:
        out_dir = project_dir
        logger = get_logger(out_dir, overwrite=True)
    
    logger("Args dump: \n{}".format(vars(args)))
    
    # Get hyperparameters
    from utime.hyperparameters import YAMLHParams
    
    if bool(args.fold):
        hpath = project_dir + "/hparams_plasma_states_fold{}.yaml".format(str(args.fold))
    elif machine == 'TCV':
        hpath = project_dir + "/hparams_plasma_states.yaml"
    elif machine == 'JET':
        hpath = project_dir + "/hparams_plasma_states_JET.yaml"

    hparams = YAMLHParams(hpath, logger)
    dataset_hparams = YAMLHParams(os.path.join(project_dir, hparams["datasets"]["dataset_1"]), logger=logger)
 
    if args.run_from_GUI:
        dataset_hparams.set_value("dataset_config", "shot", args.shot, overwrite=True)
        dataset_hparams.set_value("dataset_config", "read_csv", True, overwrite=True)
        dataset_hparams.save_current()
        if args.validate_score:
            dataset_hparams.set_value("dataset_config", "validate_score", True, overwrite=True)
            dataset_hparams.save_current()

    # Get model
    set_gpu_vis(args.num_GPUs, args.force_GPU, logger)
    model, model_func = None, None
    if args.one_shot:
        # Model is initialized for each sleep study later
        def model_func(full_states):
            return get_and_load_one_shot_model(full_states, project_dir,
                                               hparams, logger, args.fold, machine,
                                               args.weights_file_name)
    else:
        model = get_and_load_model(project_dir, hparams, logger, args.fold,
                                   machine, args.weights_file_name)
   	
    if (args.data_split == 'test_data'):
        print("To evaluate in test_data")
        shots_dir = dataset_hparams["test_data"]['data_dir']
    elif (args.data_split == 'val_data'):
        print("To evaluate in val_data")
        shots_dir = dataset_hparams["val_data"]['data_dir']
    elif (args.data_split == 'train_data'):
        print("To evaluate in train_data")
        shots_dir = dataset_hparams["train_data"]['data_dir']
    else:
        raise ValueError("Evaluation must be carried out in train, val or test data only")

    if args.run_from_GUI:
        shots_dir = os.path.abspath('./')
        shots_dir = os.path.join(shots_dir, 'data/Detected')
    
    datasets = get_splits_from_all_datasets_ps(hparams=hparams, splits_to_load=(args.data_split,),
                                    logger=logger)
    for dataset in datasets:
        dataset = dataset[0]
        run_pred_and_eval(dataset=dataset,
                  out_dir=out_dir,
                  model=model, model_func=model_func,
                  hparams=hparams,
                  args=args,
                  shots_dir=shots_dir,
                  logger=logger,
                  machine=machine)

    #if len(eval_dirs) > 1:
    #    cross_dataset_eval(eval_dirs, out_dir)


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
