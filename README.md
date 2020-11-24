# UTime-Plasma-States

Implementation of a U-Time model for time-series segmentation of plasma confinement states.

Results were published in the [Machine Learning for Physical Sciences workshop](https://ml4physicalsciences.github.io/2020/)
as part of NeurIPS 2020.

The code was adapted from:<br/>
https://github.com/perslev/U-Time<br/>
https://arxiv.org/abs/1910.11162

## Installation
<pre>
<b># Installation of Miniconda</b>
- Get and install Miniconda:
    1. `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    2. `bash Miniconda3-latest-Linux-x86_64.sh`
    3. `export PATH="/home/user/miniconda3/bin:$PATH"` (or where you have decided to install miniconda3)

<b># Install python 3.7 </b>
- `conda install -c anaconda python=3.7`

<b># GPUs </b>
- If you don't have GPUs you can skip this section.
- To run with GPUs you need to install tensorflow-gpu, do the following modification in `requirements.txt`
tensorflow==2.0 --> tensorflow-gpu==2.0
- tensorflow-gpu = 2.0 was compiled with cuda 10.0 and so if you want to use cuda 10.2 you would need to install TF from source
https://github.com/tensorflow/tensorflow/issues/34759
- After UTime installation, check if tensorflow-gpu is working:
`python -i`
`import tensorflow as tf`
`tf.test.is_gpu_available()`

<b># Installation of UTime </b>
- `git clone https://github.com/gmarceca/UTime-Plasma-States.git`
- `pip install -e UTime-Plasma-States`
- `cd UTime-Plasma-States/`

<b># Issues </b>
- In case of issues, check "Possible ISSUES" below for possible solutions.

</pre>

## Preparation of Experiments
<pre>
<b># Initialize a U-Time N-fold CV project</b>
ut init_plasmastates --name my_utime_project \
        --model utime \
        --data_dir dataset \
        --CV 1
(This prepares the settings to run all folds at once. If you want to focus on a particular fold 
pass --fold 'your_fold_number' as an additional argument)

<b># Start training</b>
cd my_utime_project
    <b># Full training (train+val / test):</b>
    ut train_plasma_states_detector --num_GPUs=1
    <b># One-fold training (train_fold / val_fold):</b>
    ut train_plasma_states_detector --num_GPUs=1 --fold='your_fold_number'
    <b># Full N-fold CV training:</b>
    `cp ../extra_scripts/run.py .`
    python run.py

</pre>

### Transfer Learning (from Sleep Staging time-series)

Most of the code was inspired from the original UTime model applied to Sleep Staging (https://github.com/perslev/U-Time).
The code structure was preserved, only doing the relevant modifications to handle our time-series data for plasma states.
Consequently, it is straightforward to do transfer learning from sleep staging data. Please find the instructions below.

#### Pre-training in the sleep staging data

A model trained in the [Sleep-EDF database](https://physionet.org/content/sleep-edf/1.0.0/) performant as in the [original paper](https://arxiv.org/abs/1910.11162) 
can be found [here](in_dir_eval/model_for_tl). You can just copy this directory to your project directory and set TransferLearning=True (default) in the hparams_plasma_states.yaml file. 

In case you want to start from scratch, run the following sections (\#) from https://github.com/perslev/U-Time:
1. \# Obtain a public sleep staging dataset
2. \# Prepare a fixed-split experiment
3. \# Initialize a U-Time project
4. Configure the hparams.yaml file accordingly and choose a model architecture. 
  The architecture selected is fixed and cannot be modified, meaning that this 
  will be the one to be used for transfer learning.
5. \# Start training (four channels):
    - `ut train --num_GPUs=1 --channels 'EEG Fpz-Cz' 'EEG Pz-Oz' 'EOG horizontal' 'EMG submental'`

### Possible ISSUES
#### Issue
- The following error was obtained in LAC from miniconda3/lib/python3.7/tkinter/:
- `RuntimeError: main thread is not in main loop`
#### Solution:
- `pip install MultiPlanarUNet`
- In `MultiPlanarUNet/utils/plotting.py` add the following lines (Inspired from https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop):
- `import matplotlib`
- `matplotlib.use('Agg')`

#### Issue
- When loading the weights from a trained model the following error was obtained:
- `AttributeError: 'str' object has no attribute 'decode'`
- https://github.com/tensorflow/tensorflow/issues/44467
#### Solution:
- Downgrade h5py-3.1.0 to h5py-2.10
