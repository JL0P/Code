"""
=======================================================================
PYTHON SCRIPT: Hyperparameter Tuning with Keras Tuner (L21 Regularization)
- Sets seeds for reproducibility
- Loads combined training data (train + val1) from a shelve file
- Defines custom L21-based regularizer for group-sparse cost-based pruning
- Builds and trains a model under different L21 hyperparameters (GridSearch)
- Saves weights and performance metrics (precision, sensitivity, etc.)
=======================================================================
"""

import os
import random
import time
import shelve
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from keras import backend as K
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow import keras
import sklearn.metrics as sk

# ==== 1. SET SEEDS FOR REPRODUCIBILITY ====
seed_value = 4

# 1.1 Environment-level seed
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 1.2 Python built-in RNG
random.seed(seed_value)
# 1.3 NumPy RNG
np.random.seed(seed_value)
# 1.4 TensorFlow RNG
tf.random.set_seed(seed_value)

# Configure TensorFlow threading
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

# ==== 3. LOAD & PREPARE SENSOR COSTS ====
PrC       = pd.read_excel('Info costiV2.xlsx', sheet_name=1)
FlC       = pd.read_excel('Info costiV2.xlsx', sheet_name=0)
flowCost  = np.array(FlC)
pressCost = np.array(PrC)

# Example: flow + pressure costs plus 0 for 'hour-of-day' cost
cost = np.concatenate((flowCost[:,4], pressCost[:,1], [0]), dtype=np.float32)

# Standardize cost array
transformer = StandardScaler().fit(cost.reshape(-1, 1))
costNorm    = transformer.transform(cost.reshape(-1, 1))

# Array of free (1) vs. paid (0) sensors
gratisFlag = np.concatenate((flowCost[:,5], pressCost[:,2], [1]), dtype=np.float32)
ind        = np.where(gratisFlag == 0)

# ==== 4. LOAD SHELVE DATA & COMBINE TRAIN/VAL1 ====
shelfIn = shelve.open(filepathShelve + testoDataset)
for key in shelfIn:
    globals()[key] = shelfIn[key]
shelfIn.close()

dataTrainInAll  = np.concatenate((dataTrainInNorm,  dataVal1InNorm), axis=0)
dataTrainOutAll = np.concatenate((dataTrainOut,     dataVal1Out),    axis=0)

# ==== 5. DEFINE INITIALIZERS & TRAINING PARAMS ====
initializerB = tf.keras.initializers.Zeros()
initializerK = tf.keras.initializers.GlorotUniform()
ep           = 150  # epochs for tuning
shuf         = True # shuffle data?

# ==== 6. CUSTOM GROUP-SPARSE L21 REGULARIZER ====
class custom_reg_builder(tf.keras.regularizers.Regularizer):
    """
    Custom group-sparse L21 regularizer that incorporates sensor cost.
    """
    def __init__(self, cost, ind, L21, L1):
        self.cost = cost
        self.ind  = ind
        self.L21  = L21
        self.L1   = L1

    def __call__(self, x):
        # Debug print: shape of the weight matrix
        print(tf.shape(x))

        # Weighted by cost, then cast
        w  = tf.math.multiply(x, self.cost)
        w  = tf.cast(w, tf.dtypes.float32)

        s  = tf.shape(x)
        sh = tf.cast(s[1], tf.dtypes.float32)

        # L1 term (commented out in final return)
        Rl1 = tf.math.multiply(self.L1, tf.reduce_sum(tf.reduce_sum(tf.abs(x),1)))

        # Weighted L21 term
        a    = tf.math.sqrt(sh) * tf.math.sqrt(tf.reduce_sum(tf.pow(w,2),1))
        out  = tf.gather(a, self.ind)
        Rl21 = tf.math.multiply(self.L21, tf.reduce_sum(out))

        # Return only L21
        return Rl21

    def get_config(self):
        """
        Make regularizer serializable by returning init args as dict.
        """
        return {
            'cost': self.cost,
            'ind':  self.ind,
            'L21':  self.L21,
            'L1':   self.L1
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ==== 7. MODEL-BUILDING FUNCTION (KERAS TUNER) ====
def model_builder(hp):
    """
    Build and compile a Keras model for L21 hyperparameter tuning.

    Args:
        hp: HyperParameters instance from Keras Tuner.

    Returns:
        A compiled Keras model.
    """
    model = keras.Sequential()

    # Tune the L21 hyperparameter in log scale from 1e-5 to ~1e-1
    L21_opt = hp.Float('L21', min_value=0.00001, max_value=0.1025, step=2, sampling="log")

    # Input layer with custom group-sparse regularizer
    model.add(tf.keras.layers.Dense(
        units=200,
        name="dense1",
        input_shape=(dataTrainInAll.shape[1],),
        kernel_regularizer=custom_reg_builder(costNorm, ind[0], L21_opt, L21_opt),
        activation='relu',
        kernel_initializer=initializerK,
        bias_initializer=initializerB
    ))

    # Hidden layer
    model.add(tf.keras.layers.Dense(
        units=300,
        name="dense2",
        activation='relu',
        kernel_initializer=initializerK,
        bias_initializer=initializerB
    ))

    # Output layer for 8-class classification
    model.add(tf.keras.layers.Dense(
        8,
        name="dense6",
        activation='softmax',
        kernel_initializer=initializerK,
        bias_initializer=initializerB
    ))

    # Optionally tune learning rate (here we fix to 1e-5)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-5])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        weighted_metrics=['categorical_accuracy']
    )
    return model

# ==== 8. KERAS TUNER SETUP (GRID SEARCH) ====
tuner = kt.GridSearch(
    hypermodel=model_builder,
    objective='val_categorical_accuracy',
    overwrite=False,
    directory=f"{projPath}/{projName}_GridSearchLambda",
    project_name="prova2",
    max_trials=None,
    seed=50,
    hyperparameters=None,
    tune_new_entries=True,
    allow_new_entries=True,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
)

# ==== 9. PREPARE TRAIN/VAL DATA & CLASS WEIGHTS ====
valW   = class_weight.compute_sample_weight('balanced', Val1Out)
trainW = class_weight.compute_sample_weight('balanced', TrainOut)

stop_early = tf.keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=10,
    restore_best_weights=True
)

# Create directory for logging if not existent
Path(infoPath + 'LOSS/' + projName).mkdir(parents=True, exist_ok=True)
csvclbck = tf.keras.callbacks.CSVLogger(infoPath + 'LOSS/' + projName + '/log.csv',
                                        separator=",", append=True)

# ==== 10. START TUNING ====
tuner.search(
    dataTrainInNorm,
    dataTrainOut,
    sample_weight=trainW,
    shuffle=shuf,
    epochs=ep,
    validation_data=(dataVal1InNorm, dataVal1Out, valW),
    callbacks=[stop_early, csvclbck]
)

# ==== 11. SAVE NUMBER OF TRIALS & EVALUATE BEST MODELS ====
Path(infoPath + f'NUM_TRIALS/{projName}').mkdir(parents=True, exist_ok=True)
numberOfTrials = len(tuner.oracle.end_order)
with open(infoPath + f'NUM_TRIALS/{projName}/nTrials.txt', 'w') as f:
    f.write(str(numberOfTrials))

modelList   = tuner.get_best_models(num_models=numberOfTrials)
model       = modelList[1]

# Example evaluation on dataVal2InNorm
y_predicted = model.predict(dataVal2InNorm)
matVal      = sk.confusion_matrix(dataVal2Out.argmax(axis=1), y_predicted.argmax(axis=1))
accVal      = matVal.diagonal() / matVal.sum(axis=1)
precisionVal= matVal.diagonal() / matVal.sum(axis=0)
avgPrecVal  = np.mean(precisionVal)
avgSensVal  = np.mean(accVal)

# ==== 12. CREATE DIRECTORIES FOR SAVING WEIGHTS/METRICS ====
Path(infoPath + 'WEIGHTS/'       + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'WEIGHTSw2/'     + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'WEIGHTSw3/'     + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'WEIGHTSwOut/'   + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'WEIGHTSb1/'     + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'WEIGHTSb2/'     + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'WEIGHTSb3/'     + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'WEIGHTSbOut/'   + projName).mkdir(parents=True, exist_ok=True)

Path(infoPath + 'Precision/'       + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'Sensitivity/'     + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'noNanPrecision/'  + projName).mkdir(parents=True, exist_ok=True)
Path(infoPath + 'noNanSensitivity/'+ projName).mkdir(parents=True, exist_ok=True)

# ==== 13. SAVE MODELS & PERFORMANCE METRICS ====
modelList = tuner.get_best_models(num_models=numberOfTrials)
for i in range(numberOfTrials):
    model = modelList[i]

    # Save best model as ID 0
    if i == 0:
        model.save(f'./{dirBestModels}/{projName}/model0.h5')
        w1 = pd.DataFrame(model.get_layer(name="dense1").get_weights()[0])
        w1.to_csv(f'./{dirBestModels}/{projName}/weight{i}.csv', header=None, index=None)

    # Save every model
    model.save(f'./{dirModels}/{projName}/model{i}.h5')

    # Export weights for each layer
    w1 = pd.DataFrame(model.get_layer(name="dense1").get_weights()[0])
    w1.to_csv(f"{infoPath}WEIGHTS/{projName}/weight{i}.csv", header=None, index=None)

    w2 = pd.DataFrame(model.get_layer(name="dense2").get_weights()[0])
    w2.to_csv(f"{infoPath}WEIGHTSw2/{projName}/weight{i}.csv", header=None, index=None)

    # If dense3 is used, handle it similarly:
    # w3 = pd.DataFrame(model.get_layer(name="dense3").get_weights()[0])
    # w3.to_csv(f"{infoPath}WEIGHTSw3/{projName}/weight{i}.csv", header=None, index=None)

    b1 = pd.DataFrame(model.get_layer(name="dense1").get_weights()[1])
    b1.to_csv(f"{infoPath}WEIGHTSb1/{projName}/weight{i}.csv", header=None, index=None)

    b2 = pd.DataFrame(model.get_layer(name="dense2").get_weights()[1])
    b2.to_csv(f"{infoPath}WEIGHTSb2/{projName}/weight{i}.csv", header=None, index=None)

    # b3 = pd.DataFrame(model.get_layer(name="dense3").get_weights()[1])
    # b3.to_csv(f"{infoPath}WEIGHTSb3/{projName}/weight{i}.csv", header=None, index=None)

    wOut = pd.DataFrame(model.get_layer(name="dense6").get_weights()[0])
    wOut.to_csv(f"{infoPath}WEIGHTSwOut/{projName}/weight{i}.csv", header=None, index=None)

    bOut = pd.DataFrame(model.get_layer(name="dense6").get_weights()[1])
    bOut.to_csv(f"{infoPath}WEIGHTSbOut/{projName}/weight{i}.csv", header=None, index=None)

    # Evaluate on Val2
    y_predicted = model.predict(dataVal2InNorm)
    matVal      = sk.confusion_matrix(dataVal2Out.argmax(axis=1), y_predicted.argmax(axis=1))
    accVal      = matVal.diagonal() / matVal.sum(axis=1)
    precisionVal= matVal.diagonal() / matVal.sum(axis=0)
    avgPrecVal  = np.mean(precisionVal)
    avgSensVal  = np.mean(accVal)
    NaNAvgPrecVal = np.nanmean(precisionVal)
    NaNAvgSensVal = np.nanmean(accVal)

    # Save metrics for Val2
    with open(f"{infoPath}Precision/{projName}/PrecVal{i}.csv",'a') as f:
        np.savetxt(f, np.array([avgPrecVal]))
    with open(f"{infoPath}Sensitivity/{projName}/SensVal{i}.csv",'a') as f:
        np.savetxt(f, np.array([avgSensVal]))
    with open(f"{infoPath}noNanPrecision/{projName}/noNanPrecVal{i}.csv",'a') as f:
        np.savetxt(f, np.array([NaNAvgPrecVal]))
    with open(f"{infoPath}noNanSensitivity/{projName}/noNanSensVal{i}.csv",'a') as f:
        np.savetxt(f, np.array([NaNAvgSensVal]))

    # Evaluate on Test
    y_predictedTest = model.predict(dataTestInNorm)
    matTest = sk.confusion_matrix(dataTestOut.argmax(axis=1), y_predictedTest.argmax(axis=1))
    accTest = matTest.diagonal() / matTest.sum(axis=1)
    precisionTest = matTest.diagonal() / matTest.sum(axis=0)
    avgPrecTest = np.mean(precisionTest)
    avgSensTest = np.mean(accTest)
    NaNAvgPrecTest = np.nanmean(precisionTest)
    NaNAvgSensTest = np.nanmean(accTest)

    # Save metrics for Test
    with open(f"{infoPath}Precision/{projName}/PrecTest{i}.csv",'a') as f:
        np.savetxt(f, np.array([avgPrecTest]))
    with open(f"{infoPath}Sensitivity/{projName}/SensTest{i}.csv",'a') as f:
        np.savetxt(f, np.array([avgSensTest]))
    with open(f"{infoPath}noNanPrecision/{projName}/noNanPrecTest{i}.csv",'a') as f:
        np.savetxt(f, np.array([NaNAvgPrecTest]))
    with open(f"{infoPath}noNanSensitivity/{projName}/noNanSensTest{i}.csv",'a') as f:
        np.savetxt(f, np.array([NaNAvgSensTest]))

# ==== 14. SAVE HYPERPARAMETERS VALUES & SCORES ====
Path(f"{infoPath}GRID/{projName}").mkdir(parents=True, exist_ok=True)

trials = tuner.oracle.get_best_trials(num_trials=numberOfTrials)
HP_list = []
for trial in trials:
    HP_list.append(
        trial.hyperparameters.get_config()["values"]
        | {"Score": trial.score}
        | {"TrialID": trial.trial_id}
        | {"Epochs": ep}
        | {"Shuffle": shuf}
    )

HP_df = pd.DataFrame(HP_list)
HP_df.to_csv(f"{infoPath}GRID/{projName}/_gridsearch.csv", index=False, na_rep='NaN')

# For debugging, check final metrics
print(HP_df)
trial.metrics.metrics['val_categorical_accuracy'].__dict__
