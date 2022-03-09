import sys
import statistics
import numpy as np
import sklearn
import seaborn as sns
import datetime
import time
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
#import pickle #use dill instead below
from statistics import mean
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten #.core
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # or Classifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
import random
#https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%202.ipynb
import random
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import collections
import operator

data = sys.argv[1]
data2 = sys.argv[2]

def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
        return (1-SS_res/SS_tot)


def unpack(model, training_config, weights): ##https://github.com/tensorflow/tensorflow/issues/34697 #fixes an error that the early stopping callback throws up in the nested cv #something about the parralele fitting step needing everything to be pickle-able and the callback isnt 
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__



x_train, y_train = load_data(data)
x_test, y_test = load_data(data2)
scaler = preprocessing.StandardScaler().fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
n_snps = x_train.shape[1]
metric_in_use = sklearn.metrics.r2_score

METRIC_ACCURACY = coeff_determination

tf.config.threading.set_inter_op_parallelism_threads(32)
tf.config.threading.set_intra_op_parallelism_threads(32)

make_keras_picklable()

learning_rates = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
scores = []
for learning_rate in learning_rates:
	opt = 'SGD'
	chosen_opt = getattr(tf.keras.optimizers,opt)
	model = Sequential()
	input_shape = (x_train.shape[1]-1,)
	model.add(Dense(units=100, activation='relu', input_shape=input_shape))
	model.add(Dense(units=100, activation='relu')
	model.add(Dense(1, activation='linear'))
	model.compile(loss='mean_absolute_error',metrics=['mae'],optimizer=chosen_opt(learning_rate=learning_rate))
	model.fit(x_train, y_train, validation_data =(x_test, y_test), batch_size=32, epochs=200, verbose=2)
        score.append(model.score(x_test, y_test))


import matplotlib.pyplot as plt
plt.plot(learning_rates, scores)
plt.xscale("log")
plt.savefig("log_lr_scores_tibia")
plt.clf(); plt.close()


