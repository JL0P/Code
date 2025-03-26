"""
===========================================================
PYTHON SCRIPT: Hyperparameter Tuning with Keras Tuner (GridSearch)
- Sets seeds for reproducibility
- Loads data from shelve file
- Builds a neural network model with variable layers/units
- Performs a grid search over multiple hyperparameters
- Logs results and saves them for later usage
===========================================================
"""

import os
import random
import time
import shelve
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import keras_tuner as kt
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow import keras

# ==== 1. SET SEEDS FOR REPRODUCIBILITY ====
seed_value = 4

# 1.1 Fix environment-level seed
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 1.2 Python built-in RNG
random.seed(seed_value)
# 1.3 NumPy RNG
np.random.seed(seed_value)
# 1.4 TensorFlow RNG
tf.random.set_seed(seed_value)

# Configure threading (optional, for reproducibility/performance)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ==== 2. PROJECT DIRECTORIES & NAMES ====
projName     = 'projName'
testoDataset = 'dataset.out'
filepathShelve= './SHELF/'
dirModels    = 'TUNED_MODELS'
dirBestModels= 'BEST_MODELS'
projPath     = './TUNER_RESULTS'
infoPath     = './INFO/'

# ==== 3. LOAD SENSOR COSTS ====
PrC         = pd.read_excel('Info costiV2.xlsx', sheet_name=1)
FlC         = pd.read_excel('Info costiV2.xlsx', sheet_name=0)
flowCost    = np.array(FlC)
pressCost   = np.array(PrC)

# Example: combine flow + pressure costs, plus an additional 0 for hour-of-day
cost        = np.concatenate((flowCost[:, 4], pressCost[:, 1], [0]), dtype=np.float32)

# Standardize cost array
transformer = StandardScaler().fit(cost.reshape(-1, 1))
costNorm    = transformer.transform(cost.reshape(-1, 1))

# Check which sensors are free or not
gratisFlag = np.concatenate((flowCost[:, 5], pressCost[:, 2], [1]), dtype=np.float32)
ind        = np.where(gratisFlag == 0)

# ==== 4. LOAD SHELVE DATA (DATASET) ====
shelfIn = shelve.open(filepathShelve + testoDataset)
for key in shelfIn:
    globals()[key] = shelfIn[key]   # Bring variables into global scope
shelfIn.close()

# ==== 5. DEFINE INITIALIZERS & TRAINING PARAMS ====
initializerB = tf.keras.initializers.Zeros()
initializerK = tf.keras.initializers.GlorotUniform()
ep           = 150   # Number of epochs for tuning
shuf         = True  # Shuffle data?

# ==== 6. MODEL-BUILDING FUNCTION FOR KERAS TUNER ====
def model_builder(hp):
    """
    Build and compile a Keras model, with hyperparameters tuned by Keras Tuner.

    Args:
        hp: HyperParameters instance from Keras Tuner.

    Returns:
        Compiled tf.keras.Model.
    """
    model = keras.Sequential()

    # First layer: number of units can be 50, 100, 200, or 300
    model.add(
        tf.keras.layers.Dense(
            units = hp.Choice('units00', values=[50, 100, 200, 300]),
            name="dense1",
            input_shape=(len(dataTrainInNorm.T),),
            activation='relu',
            kernel_initializer=initializerK,
            bias_initializer=initializerB
        )
    )

    # Optionally add more layers if hp.Boolean("MoreLayers") is True
    if hp.Boolean("MoreLayers"):
        # The number of extra layers is between 1 and 2
        for i in range(hp.Int("num_layers", 1, 2, step=1)):
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Choice(f'units_{i}', values=[50, 100, 200, 300]),
                    activation='relu',
                    kernel_initializer=initializerK,
                    bias_initializer=initializerB
                )
            )

    # Output layer: 8 classes, softmax activation
    model.add(
        tf.keras.layers.Dense(
            8, 
            name="dense6", 
            activation='softmax',
            kernel_initializer=initializerK,
            bias_initializer=initializerB
        )
    )

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-6])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        weighted_metrics=['categorical_accuracy']
    )

    return model

# ==== 7. SETUP KERAS TUNER (GRID SEARCH) ====
tuner = kt.GridSearch(
    hypermodel=model_builder,
    objective='val_categorical_accuracy',
    overwrite=False,  # If True, previous tuning results are discarded
    directory= f"{projPath}/{projName}_GridSearch", 
    project_name="prova",
    max_trials=None,  # No explicit limit on trials
    seed=50,
    hyperparameters=None,
    tune_new_entries=True,
    allow_new_entries=True,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
)

# ==== 8. PREPARE TRAIN/VAL DATA & CLASS WEIGHTS ====
valW   = class_weight.compute_sample_weight('balanced', Val1Out)
trainW = class_weight.compute_sample_weight('balanced', TrainOut)

stop_early = tf.keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=10,
    restore_best_weights=True
)

# Create directory for logging if not existent
Path(infoPath+'LOSS/'+projName).mkdir(parents=True, exist_ok=True)
csvclbck = tf.keras.callbacks.CSVLogger(infoPath+'LOSS/'+projName+'/log.csv', separator=",", append=True)

# ==== 9. START TUNING ====
tuner.search(
    dataTrainInNorm, 
    dataTrainOut, 
    sample_weight=trainW,
    shuffle=shuf,
    epochs=ep,
    validation_data=(dataVal1InNorm, dataVal1Out, valW),
    callbacks=[stop_early, csvclbck]
)

# ==== 10. SAVE TUNING RESULTS & TRIAL COUNTS ====
Path(infoPath+'NUM_TRIALS/'+projName).mkdir(parents=True, exist_ok=True)
numberOfTrials = len(tuner.oracle.end_order)
with open(infoPath + f'NUM_TRIALS/{projName}/nTrials.txt', 'w') as f:
    f.write(f'{numberOfTrials}')

tuner.results_summary()

# ==== 11. SAVE HYPERPARAMETERS (GRID SEARCH) TO CSV ====
Path(infoPath+'GRID/'+projName).mkdir(parents=True, exist_ok=True)

# Retrieve all trials (best to worst)
trials = tuner.oracle.get_best_trials(num_trials=numberOfTrials)

HP_list = []
HP_list = []
for trial in trials:
    HP_list.append(trial.hyperparameters.get_config()["values"] | {"Score": trial.score} | {'TrialID':trial.trial_id} | {'Epochs':ep} | {'Shuffle':shuf})
HP_df = pd.DataFrame(HP_list)
HP_df.to_csv(infoPath+'GRID/'+projName+"/_gridsearch.csv", index=False, na_rep='NaN')