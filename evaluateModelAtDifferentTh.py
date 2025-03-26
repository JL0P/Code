### === 1. IMPORTS & ENV SETUP ==
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import shelve
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import sklearn.metrics as sk

# Fix random seed for reproducibility
seed_value = 4
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


### === 2. PATHS AND CONFIG ===
testoDataset = 'dataset.out'
filepathShelve = './SHELF/'
dirModels = 'TUNED_MODELS'
dirBestModels = 'BEST_MODELS'
projPath = './TUNER_RESULTS'
infoPath = './INFO/'

modelType = 'l21l1l12'


### === 3. LOAD DATA ===
shelfIn = shelve.open(filepathShelve + testoDataset)
data_keys = [
    'dataTrainInNorm', 'dataTrainOut', 'dataVal1InNorm', 'dataVal1Out',
    'dataVal2InNorm', 'dataVal2Out', 'dataTestInNorm', 'dataTestOut',
    'd_class_weights', 'Val2Out', 'TrainOut'
]
for key in data_keys:
    if key in shelfIn:
        globals()[key] = shelfIn[key]
shelfIn.close()
del shelfIn


### === 4. LOAD GRID PARAMETERS ===
grid = np.genfromtxt(f'./INFO/GRID/{projName}/_gridsearch.csv', delimiter=',', skip_header=1)
gridTxt = np.genfromtxt(f'./INFO/GRID/{projName}/_gridsearch.csv', delimiter=',', skip_header=1, dtype=str)

lr = float(grid[modelNum, 1])
ep = maxEpochs
L21 = float(grid[modelNum, 0])
shuf = gridTxt[modelNum, 5]

print(grid[modelNum, -1])
del grid, gridTxt


### === 5. LOAD COSTS AND NORMALIZE ===
PrC = pd.read_excel('Info costiV2.xlsx', sheet_name=1)
FlC = pd.read_excel('Info costiV2.xlsx', sheet_name=0)
flowCost = np.array(FlC)
pressCost = np.array(PrC)
del PrC, FlC

cost = np.concatenate((flowCost[:, 4], pressCost[:, 1], [300]), dtype=np.float32)
gratisFlag = np.concatenate((flowCost[:, 5], pressCost[:, 2], [1]), dtype=np.float32)
ind = np.where(gratisFlag == 0)

transformer = StandardScaler().fit(cost.reshape(-1, 1))
costNorm = transformer.transform(cost.reshape(-1, 1))


### === 6. COMBINE TRAIN AND VAL1 DATA ===
dataTrainInAll = np.concatenate((dataTrainInNorm, dataVal1InNorm), axis=0)
dataTrainOutAll = np.concatenate((dataTrainOut, dataVal1Out), axis=0)


### === 7. CUSTOM REGULARIZER ===
class custom_reg_builder(tf.keras.regularizers.Regularizer):
    def __init__(self, cost, ind, L21, L1):
        self.cost = cost
        self.ind = ind
        self.L21 = L21
        self.L1 = L1

    def __call__(self, x):
        w = tf.math.multiply(x, self.cost)
        w = tf.cast(w, tf.float32)
        sh = tf.cast(tf.shape(x)[1], tf.float32)

        Rl1 = self.L1 * tf.reduce_sum(tf.reduce_sum(tf.abs(x), 1))
        a = tf.sqrt(sh) * tf.sqrt(tf.reduce_sum(tf.pow(w, 2), 1))
        out = tf.gather(a, self.ind)
        Rl21 = self.L21 * tf.reduce_sum(out)

        return Rl21

    def get_config(self):
        return {'cost': self.cost, 'ind': self.ind, 'L21': self.L21, 'L1': self.L1}


### === 8. CUSTOM CONSTRAINT TO FIX MASKED WEIGHTS ===
class FixWeights(tf.keras.constraints.Constraint):
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


### === 9. INITIALIZE MODEL WEIGHTS ===
initw1 = tf.constant_initializer(np.float32(w1))
initw2 = tf.constant_initializer(np.float32(w2))
initwOut = tf.constant_initializer(np.float32(wOut))

initb1 = tf.constant_initializer(np.float32(b1))
initb2 = tf.constant_initializer(np.float32(b2))
initbOut = tf.constant_initializer(np.float32(bOut))

mask = np.float32(mask)
segno1 = np.float32(segno1)


### === 10. MODEL DEFINITION ===
model = tf.keras.Sequential([
    tf.keras.layers.Dense(200, activation='relu', name='dense1',
        input_shape=(dataTrainInAll.shape[1],),
        kernel_initializer=initw1, bias_initializer=initb1,
        kernel_regularizer=custom_reg_builder(costNorm, ind[0], L21, L21),
        kernel_constraint=FixWeights(mask, segno1)),

    tf.keras.layers.Dense(300, activation='relu', name='dense2',
        kernel_initializer=initw2, bias_initializer=initb2),

    tf.keras.layers.Dense(8, activation='softmax', name='dense6',
        kernel_initializer=initwOut, bias_initializer=initbOut)
])

model.summary()
model.run_eagerly = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    weighted_metrics=['categorical_accuracy']
)


### === 11. TRAINING ===
stop_early = tf.keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy', patience=10, restore_best_weights=True)

if flagMask == 1:  # Optional sensor removal
    mask[[6, 8, 13, 14]] = 0

valW2 = class_weight.compute_sample_weight('balanced', Val2Out)
trainWAll = class_weight.compute_sample_weight('balanced', dataTrainOutAll)

history = model.fit(
    dataTrainInAll * mask,
    dataTrainOutAll,
    sample_weight=trainWAll,
    epochs=ep,
    shuffle=True,
    validation_data=(dataVal2InNorm * mask, dataVal2Out, valW2),
    validation_freq=1,
    callbacks=[]
)


### === 12. CLEANUP & SAVE HISTORY ===
del initw1, initw2, initwOut, initb1, initb2, initbOut, segno1

del dataTrainInAll, dataTrainOutAll

gc.collect()

loss = history.history['loss']
categorical_accuracy = history.history['categorical_accuracy']
val_loss = history.history['val_loss']
val_categorical_accuracy = history.history['val_categorical_accuracy']


### === 13. EVALUATION ON TEST SET ===
y_predictedTest = model.predict(dataTestInNorm * mask)
matTest = sk.confusion_matrix(dataTestOut.argmax(axis=1), y_predictedTest.argmax(axis=1))
accTest = matTest.diagonal() / matTest.sum(axis=1)
precisionTest = matTest.diagonal() / matTest.sum(axis=0)

avgPrecTest = np.mean(precisionTest)
avgSensTest = np.mean(accTest)
NaNAvgPrecTest = np.nanmean(precisionTest)
NaNAvgSensTest = np.nanmean(accTest)

m = tf.keras.metrics.CategoricalAccuracy()
m.update_state(dataTestOut, y_predictedTest)
test_categorical_accuracy = m.result().numpy()


### === 14. EVALUATION ON VALIDATION SET ===
y_predictedVal = model.predict(dataVal2InNorm * mask)
matVal = sk.confusion_matrix(dataVal2Out.argmax(axis=1), y_predictedVal.argmax(axis=1))
accVal = matVal.diagonal() / matVal.sum(axis=1)
precisionVal = matVal.diagonal() / matVal.sum(axis=0)

avgPrecVal = np.mean(precisionVal)
avgSensVal = np.mean(accVal)
NaNAvgPrecVal = np.nanmean(precisionVal)
NaNAvgSensVal = np.nanmean(accVal)


### === 15. FINAL CLEANUP ===
del dataTrainInNorm, dataTrainOut, dataTestInNorm, dataTestOut

del dataVal2InNorm, dataVal2Out, dataVal1InNorm, dataVal1Out

del d_class_weights, Val2Out, TrainOut, mask

gc.collect()
