"""
==============================================================
PYTHON SCRIPT: Training & Evaluating a TensorFlow Model with RL21 Regularization
- This script loads training/validation/test data from a shelve file.
- Sets seeds for reproducibility.
- Prepares a custom regularizer and mask constraint.
- Builds and trains a model with given initial weights.
- Evaluates on test/validation sets and prints confusion matrices.
- Saves the model weights for later reuse.
==============================================================
"""

# ==== 1. IMPORTS & ENVIRONMENT SETUP ====
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import shelve
import gc
import sklearn.metrics as sk
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# --- 1.1 Set random seed for reproducibility ---
seed_value = 4
os.environ['PYTHONHASHSEED'] = str(seed_value)  # Environment-level seed
random.seed(seed_value)                        # Python built-in RNG
np.random.seed(seed_value)                     # NumPy RNG
tf.random.set_seed(seed_value)                 # TensorFlow RNG
# For older TF versions: tf.compat.v1.set_random_seed(seed_value)

# ==== 2. PROJECT DIRECTORIES & GLOBAL CONFIG ====
testoDataset   = 'dataset.out'    # Shelve dataset file
filepathShelve = './SHELF/'                           # Shelve directory
dirModels      = 'TUNED_MODELS'                       # Where to save all trial models
dirBestModels  = 'BEST_MODELS'                        # Where to save best models
projPath       = './TUNER_RESULTS'                    # Tuner raw output
infoPath       = './INFO/'                            # For saving weights & grid search data

modelType = 'l21l1l12'  # Example: model types, e.g. 'l21l1', 'l21l12', etc.

# If you need these:
# projName = 'provaL21L1_L12_senzaEarly_ShufTrue_withCost_2'
# modelNum = 0  # 0 => best model, or in [0..numOfTrials-1]

# ==== 3. LOAD DATA FROM SHELVE ====
shelfIn = shelve.open(filepathShelve + testoDataset)

for key in shelfIn:
    # We only load certain keys into global scope
    if key in [
        'dataTrainInNorm', 'dataTrainOut', 
        'dataVal2InNorm', 'dataVal2Out',
        'dataVal1InNorm', 'dataVal1Out',
        'dataTestInNorm','dataTestOut',
        'd_class_weights', 'Val2Out', 'TrainOut'
    ]:
        globals()[key] = shelfIn[key]

shelfIn.close()
del shelfIn

# ==== 4. INITIALIZE HYPERPARAMETERS & MODEL WEIGHTS ====
initializerB = tf.keras.initializers.Zeros()
initializerK = tf.keras.initializers.GlorotUniform()

# Load grid search CSV
grid    = np.genfromtxt('./INFO/GRID/' + projName + '/_gridsearch.csv',
                        delimiter=',', skip_header=1, dtype=np.float_)
gridTxt = np.genfromtxt('./INFO/GRID/' + projName + '/_gridsearch.csv',
                        delimiter=',', skip_header=1, dtype=str)
print(grid[modelNum, -1])  # Print some final param from the grid

# Extract hyperparameters from the grid
lr   = float(grid[modelNum, 1])    # Learning rate
ep   = maxEpochs                   # Number of epochs
L21  = float(grid[modelNum, 0])    # L21 reg
shuf = gridTxt[modelNum, 5]        # Shuffle setting
del grid, gridTxt

# Merge training data with val1 data
dataTrainInAll  = np.concatenate((dataTrainInNorm,  dataVal1InNorm), axis=0)
dataTrainOutAll = np.concatenate((dataTrainOut,     dataVal1Out),    axis=0)

# ==== 5. LOAD & NORMALIZE COSTS ====
PrC      = pd.read_excel('Info costiV2.xlsx', sheet_name=1)
FlC      = pd.read_excel('Info costiV2.xlsx', sheet_name=0)
flowCost  = np.array(FlC)
pressCost = np.array(PrC)
del PrC, FlC

# Combine cost arrays (flow + pressure + a synthetic time cost)
cost       = np.concatenate((flowCost[:,4], pressCost[:,1], [300]), dtype=np.float32)
gratisFlag = np.concatenate((flowCost[:,5], pressCost[:,2], [1]),   dtype=np.float32)
ind = np.where(gratisFlag == 0)

# Standardize cost
transformer = StandardScaler().fit(cost.reshape(-1, 1))
costNorm    = transformer.transform(cost.reshape(-1,1))

# ==== 6. CUSTOM REGULARIZER & CONSTRAINT CLASSES ====
class custom_reg_builder(tf.keras.regularizers.Regularizer):
    """
    Custom L21-based regularizer with cost weighting.
    """
    def __init__(self, cost, ind, L21, L1):
        self.cost = cost
        self.ind = ind
        self.L21 = L21
        self.L1 = L1

    def __call__(self, x):
        print(tf.shape(x))  # Debug: shape of the weight matrix
        w = tf.math.multiply(x, self.cost)
        w = tf.cast(w, tf.float32)
        s = tf.shape(x)
        sh = tf.cast(s[1], tf.float32)

        # L1 portion
        Rl1 = tf.math.multiply(self.L1, tf.reduce_sum(tf.reduce_sum(tf.abs(x),1)))
        # Weighted L21 portion
        a = tf.math.sqrt(sh) * tf.math.sqrt(tf.reduce_sum(tf.pow(w,2),1))
        out = tf.gather(a, self.ind)
        Rl21 = tf.math.multiply(self.L21, tf.reduce_sum(out))

        # Return only the L21 portion (L1 portion is commented out)
        return Rl21

    def get_config(self):
        """
        Make this serializable by returning init args as dictionary.
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

class FixWeights(tf.keras.constraints.Constraint):
    """
    Constraint class for forcing certain weights to a near-zero constant,
    effectively masking them.
    """
    def __init__(self, my_constraints, segno1):
        self.my_constraints = my_constraints
        self.segno1 = segno1

    def __call__(self, w):
        for i in range(len(self.my_constraints)):
            if self.my_constraints[i] == 0:
                w[i].assign(self.segno1[i] * 2.2204e-16)   
        return w

    def get_config(self):
        return {'my_constraints': self.my_constraints}

# ==== 7. INIT MODEL & LAYERS ====
initw1   = tf.constant_initializer(np.float32(w1))
initw2   = tf.constant_initializer(np.float32(w2))
# initw3 = tf.constant_initializer(np.float32(w3))
initwOut = tf.constant_initializer(np.float32(wOut))

initb1   = tf.constant_initializer(np.float32(b1))
initb2   = tf.constant_initializer(np.float32(b2))
# initb3 = tf.constant_initializer(np.float32(b3))
initbOut = tf.constant_initializer(np.float32(bOut))

mask   = np.float32(mask)
segno1 = np.float32(segno1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        200, 
        name="dense1",
        input_shape=(dataTrainInAll.shape[1],),
        kernel_constraint=FixWeights(mask, segno1),
        kernel_regularizer=custom_reg_builder(costNorm, ind[0], L21, L21),
        activation='relu',
        kernel_initializer=initw1,
        bias_initializer=initb1
    ),
    tf.keras.layers.Dense(
        300,
        name="dense2",
        activation='relu',
        kernel_initializer=initw2,
        bias_initializer=initb2
    ),
    # tf.keras.layers.Dense(120, name="dense3", activation='relu', kernel_initializer=initw3, bias_initializer=initb3),
    tf.keras.layers.Dense(
        8,
        name="dense6",
        activation='softmax',
        kernel_initializer=initwOut,
        bias_initializer=initbOut
    )
])

model.summary()
model.run_eagerly = True  # Debugging: track ops in eager mode

# Compile with categorical crossentropy and weighted accuracy
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    weighted_metrics=['categorical_accuracy']
)

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=10,
    restore_best_weights=True
)

# Optionally zero out certain mask indices
if flagMask == 1:
    mask[6]  = 0
    mask[8]  = 0
    mask[13] = 0
    mask[14] = 0

# Compute sample weights for balanced classes
valW2     = class_weight.compute_sample_weight('balanced', Val2Out)  
trainWAll = class_weight.compute_sample_weight('balanced', dataTrainOutAll)

# ==== 8. TRAIN THE MODEL ====
history = model.fit(
    dataTrainInAll * mask,
    dataTrainOutAll,
    epochs=ep,
    shuffle=shuf,
    # class_weight=d_class_weights,  # optional
    callbacks=[],
    validation_batch_size=None,
    validation_data=(dataVal2InNorm*mask, dataVal2Out),
    validation_freq=1
)

# (Optional) Save initial layer weights to CSV if needed:
# np.savetxt("provaW.csv", model.layers[0].get_weights()[0], delimiter=';')

# Create path to store model weights
Path('./MODELS_WITH_MASK/' + projName + '/trainedMODELSweigths').mkdir(parents=True, exist_ok=True)

# Save model weights
model.save('./MODELS_WITH_MASK/' + projName + '/trainedMODELSweigths/model' + str(modelNum) + '_weights.h5')

# ==== 9. RELOAD WEIGHTS INTO A NEW MODEL FOR VERIFICATION ====
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(200, name="dense1", input_shape=(dataTrainInAll.shape[1],), activation='relu'),
    tf.keras.layers.Dense(300, name="dense2", activation='relu'),
    # tf.keras.layers.Dense(120, name="dense3", activation='relu'),
    tf.keras.layers.Dense(8,   name="dense6", activation='softmax')
])
model2.summary()
model2.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    weighted_metrics=['categorical_accuracy']
)

# Load the saved weights
model2.load_weights('./MODELS_WITH_MASK/' + projName + '/trainedMODELSweigths/model' + str(modelNum) + '_weights.h5')

# Save entire model (architecture + weights)
Path('./MODELS_WITH_MASK/' + projName + '/trainedMODELSweigths').mkdir(parents=True, exist_ok=True)
model2.save('./MODELS_WITH_MASK/' + projName + '/trainedMODELS/model' + str(modelNum) + '.h5')

# ==== 10. CLEANUP & GARBAGE COLLECTION ====
del initw1, initw2, initwOut
del initb1, initb2, initbOut
del segno1
gc.collect()

# Extract training metrics
loss                  = history.history['loss']
categorical_accuracy  = history.history['categorical_accuracy']
val_loss              = history.history['val_loss']
val_categorical_accuracy = history.history['val_categorical_accuracy']

# ==== 11. EVALUATE ON TEST SET ====
y_predictedTest = model.predict(dataTestInNorm * mask)
matTest   = sk.confusion_matrix(dataTestOut.argmax(axis=1), y_predictedTest.argmax(axis=1))
accTest   = matTest.diagonal() / matTest.sum(axis=1)
precisionTest = matTest.diagonal() / matTest.sum(axis=0)

avgPrecTest   = np.mean(precisionTest)
avgSensTest   = np.mean(accTest)
NaNAvgPrecTest = np.nanmean(precisionTest)
NaNAvgSensTest = np.nanmean(accTest)

m = tf.keras.metrics.CategoricalAccuracy()
m.update_state(dataTestOut, y_predictedTest)
test_categorical_accuracy = m.result().numpy()

print('\n\nSENSITIVITY:')
print('NoLeaks = '+str(accTest[0]*100)+'%\nCL1 = '+str(accTest[1]*100)+'%\nCL2 = '+str(accTest[2]*100)+\
      '%\nCL3 = '+str(accTest[3]*100)+'%\nCL4 = '+str(accTest[4]*100)+'%\nCL5 = '+str(accTest[5]*100)+\
      '%\nCL6 = '+str(accTest[6]*100)+'%\nCL7 = '+str(accTest[7]*100)+'%')

print('\n\nPRECISION:')
print('NoLeaks = '+str(precisionTest[0]*100)+'%\nCL1 = '+str(precisionTest[1]*100)+'%\nCL2 = '+\
      str(precisionTest[2]*100)+'%\nCL3 = '+str(precisionTest[3]*100)+'%\nCL4 = '+\
      str(precisionTest[4]*100)+'%\nCL5 = '+str(precisionTest[5]*100)+'%\nCL6 = '+\
      str(precisionTest[6]*100)+'%\nCL7 = '+str(precisionTest[7]*100)+'%')

print(avgPrecTest)
print(avgSensTest)

# ==== 12. EVALUATE ON VALIDATION SET (model) ====
y_predictedVal = model.predict(dataVal2InNorm * mask)
matVal   = sk.confusion_matrix(dataVal2Out.argmax(axis=1), y_predictedVal.argmax(axis=1))
accVal   = matVal.diagonal() / matVal.sum(axis=1)
precisionVal = matVal.diagonal() / matVal.sum(axis=0)

avgPrecVal   = np.mean(precisionVal)
avgSensVal   = np.mean(accVal)
NaNAvgPrecVal = np.nanmean(precisionVal)
NaNAvgSensVal = np.nanmean(accVal)

print('\n\nSENSITIVITY:')
print('NoLeaks = '+str(accVal[0]*100)+'%\nCL1 = '+str(accVal[1]*100)+'%\nCL2 = '+str(accVal[2]*100)+\
      '%\nCL3 = '+str(accVal[3]*100)+'%\nCL4 = '+str(accVal[4]*100)+'%\nCL5 = '+str(accVal[5]*100)+\
      '%\nCL6 = '+str(accVal[6]*100)+'%\nCL7 = '+str(accVal[7]*100)+'%')

print('\n\nPRECISION:')
print('NoLeaks = '+str(precisionVal[0]*100)+'%\nCL1 = '+str(precisionVal[1]*100)+'%\nCL2 = '+\
      str(precisionVal[2]*100)+'%\nCL3 = '+str(precisionVal[3]*100)+'%\nCL4 = '+\
      str(precisionVal[4]*100)+'%\nCL5 = '+str(precisionVal[5]*100)+'%\nCL6 = '+\
      str(precisionVal[6]*100)+'%\nCL7 = '+str(precisionVal[7]*100)+'%')

print('model1')
print(avgPrecVal)
print(avgSensVal)

# ==== 13. EVALUATE ON VALIDATION SET (model2) ====
y_predictedVal2 = model2.predict(dataVal2InNorm*mask)
matVal2 = sk.confusion_matrix(dataVal2Out.argmax(axis=1), y_predictedVal2.argmax(axis=1))
accVal2 = matVal2.diagonal() / matVal2.sum(axis=1)
precisionVal2 = matVal2.diagonal() / matVal2.sum(axis=0)

avgPrecVal2   = np.mean(precisionVal2)
avgSensVal2   = np.mean(accVal2)
NaNAvgPrecVal2 = np.nanmean(precisionVal2)
NaNAvgSensVal2 = np.nanmean(accVal2)

print('\n\nSENSITIVITY:')
print('NoLeaks = '+str(accVal2[0]*100)+'%\nCL1 = '+str(accVal2[1]*100)+'%\nCL2 = '+str(accVal2[2]*100)+\
      '%\nCL3 = '+str(accVal2[3]*100)+'%\nCL4 = '+str(accVal2[4]*100)+'%\nCL5 = '+str(accVal2[5]*100)+\
      '%\nCL6 = '+str(accVal2[6]*100)+'%\nCL7 = '+str(accVal2[7]*100)+'%')

print('\n\nPRECISION:')
print('NoLeaks = '+str(precisionVal2[0]*100)+'%\nCL1 = '+str(precisionVal2[1]*100)+'%\nCL2 = '+\
      str(precisionVal2[2]*100)+'%\nCL3 = '+str(precisionVal2[3]*100)+'%\nCL4 = '+\
      str(precisionVal2[4]*100)+'%\nCL5 = '+str(precisionVal2[5]*100)+'%\nCL6 = '+\
      str(precisionVal2[6]*100)+'%\nCL7 = '+str(precisionVal2[7]*100)+'%')

print('model2')
print(avgPrecVal2)
print(avgSensVal2)

# (Optional) Delete references if you want to free memory
del mask
del dataTrainInNorm, dataTrainOut
del dataTestInNorm, dataTestOut
del dataVal2InNorm, dataVal2Out
del dataVal1InNorm, dataVal1Out
del d_class_weights, Val2Out, TrainOut

gc.collect()
