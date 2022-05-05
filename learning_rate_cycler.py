import sys; sys.path.insert(0, '/home/hers_en/rmclaughlin/tf/lib/python3.6/site-packages') ; sys.path.insert(0, '/hpc/local/CentOS7/modulefiles/python_libs/3.6.1'); sys.path.insert(0, '/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/envciaran2/lib/python3.6/site-packages')
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


organism = 'mouse'
set_size = 20000
print(set_size)
def load_data(data):
        dataset = np.loadtxt(data, skiprows=1, dtype='str')
        x = dataset[: , 6:set_size+6].astype(np.int) if organism != 'Arabadopsis' else dataset[: , 6:set_size+6]/2 #Arabdopsis data is inbred to homozyotisity to be 2/0
        y = dataset[: , 5 ].astype(np.float)
        y = y.reshape(-1,1)
        #print("Performing split of raw data....")
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
        return x, y #x_train, y_train, x_test, y_test

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


import sklearn.model_selection
x_train, y_train = load_data(data)
x_test, y_test = load_data(data2)

##warning doing split!!
#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_train[:,:], y_train[:], test_size=0.15, random_state=42)

scaler = preprocessing.StandardScaler().fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)

n_snps = x_train.shape[1]
metric_in_use = sklearn.metrics.r2_score

METRIC_ACCURACY = coeff_determination

tf.config.threading.set_inter_op_parallelism_threads(32)
tf.config.threading.set_intra_op_parallelism_threads(32)

make_keras_picklable()

#learning_rates = [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
learning_rates = [0.0005, 0.0001, 0.00005, 0.00001, 0.000001, 0.0000001,0.00000001] ; print(learning_rates)


l1_reg = [0, 0.1, 0.01, 0.2, 0.5 ,1] ; print(l1_reg)
l2_reg = [0.001, 0.00001, 0.000001] ; print(l2_reg)
dropout = [0.1, 0.3, 0.5, 0.7] ; print(dropout)
epochs = [100, 5000, 1000] ; print(epochs)
units = [100,900] ; print(units)
import random
scores = []
train_mae = []
train_r2 =[] 
r2 = []

#for a in learning_rates: 
for i in range(1,200):
	a = random.choice(learning_rates)
	b = random.choice(l1_reg)
	c = random.choice(l2_reg)
	d = random.choice(dropout)
	e = random.choice(epochs)
	f = random.choice(units)
	print(a,b,c,d,e,f, sep=',')
	opt = 'SGD'
	chosen_opt = getattr(tf.keras.optimizers,opt)
	model = Sequential()
	input_shape = (x_train.shape[1],)
	model.add(Dense(units=f, activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=b,l2=c)))
	model.add(Dropout(d))
	model.add(Dense(units=f, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=b,l2=c)))
	model.add(Dropout(d))
	model.add(Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=b,l2=c)))
	model.compile(loss='mean_squared_error',metrics=['mse', coeff_determination],optimizer=chosen_opt(learning_rate=a))
	model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=e, verbose=2)
	scores.append(model.evaluate(x_test, y_test)[0])
	train_mae.append(model.evaluate(x_train, y_train)[0])
	train_r2.append(model.evaluate(x_train, y_train)[2])
	r2.append(model.evaluate(x_test, y_test)[2])
	print(model.evaluate(x_test, y_test)[2])
	del(model)


print(scores)
print(train_mae)
print(train_r2)
print(r2)
import matplotlib.pyplot as plt
plt.plot(learning_rates, scores)
plt.xscale("log")
plt.savefig("log_lr_scores_startle")
plt.clf(); plt.close()
