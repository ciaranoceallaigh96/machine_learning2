import sys
import random
import statistics
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
import time
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
#import pickle #use dill instead below
from statistics import mean
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import make_scorer
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten #.core
from tensorflow.keras.optimizers import SGD
from sklearn.externals import joblib
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # or Classifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import math

def load_data(data):
        dataset = np.loadtxt(data, skiprows=1)
        x = dataset[: , 6:set_size+6]/2
        y = dataset[: , 5 ]
        y = y.reshape(-1,1)
        #print("Performing split of raw data....")
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
        return x, y #x_train, y_train, x_test, y_test

date_object = datetime.datetime.now().replace(second=0,microsecond=0)
print(date_object)

def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
        return (1-SS_res/SS_tot)



set_size = int(sys.argv[4])
X_train, y_train = load_data(str(sys.argv[1]))
X_test, y_test = load_data(str(sys.argv[2]))
scaler = preprocessing.StandardScaler().fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)





beta_weights = []

with open(str(sys.argv[3]), 'r') as f:
	for line in f:
		beta_weights.append(float(line.strip()))

param_grid = {'epochs' : [50,100],'batch_size' : [16,64, 128],'learning_rate' : [0.01, 0.001, 0.0001, 0.00001],'HP_L1_REG' : [1e-4, 1e-2, 1e-5,1e-6],'HP_L2_REG' : [1e-8, 0.1, 1e-4, 1e-2], 'kernel_initializer' : ['glorot_uniform', 'glorot_normal', 'he_normal'],'activation' : ['tanh'],'HP_NUM_HIDDEN_LAYERS' : [2,3],'units' : [200, 100], 'rate' : [float(0), 0.1, 0.5],'HP_OPTIMIZER' : ['Adam', 'SGD', 'Adagrad']}

METRIC_ACCURACY = coeff_determination
tf.config.threading.set_inter_op_parallelism_threads(64)
tf.config.threading.set_intra_op_parallelism_threads(64)

count = 20
while count > 0:

	epochs = random.choice(param_grid['epochs'])
	batch_size= random.choice(param_grid['batch_size'])
	learning_rate= random.choice(param_grid['learning_rate'])
	HP_L1_REG= random.choice(param_grid['HP_L1_REG'])
	HP_L2_REG= random.choice(param_grid['HP_L2_REG'])
	kernel_initializer= random.choice(param_grid['kernel_initializer'])
	activation= random.choice(param_grid['activation'])
	HP_NUM_HIDDEN_LAYERS= random.choice(param_grid['HP_NUM_HIDDEN_LAYERS'])
	units= random.choice(param_grid['units'])
	rate= random.choice(param_grid['rate'])
	HP_OPTIMIZER= random.choice(param_grid['HP_OPTIMIZER'])

	opt = HP_OPTIMIZER; chosen_opt = getattr(tf.keras.optimizers,opt)
	reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
	model = Sequential()
	model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer, input_shape=(X_train.shape[1],)))
	if rate != 0:
		model.add(Dropout(rate=rate))
	for i in range(HP_NUM_HIDDEN_LAYERS-1):
		model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
		if rate != 0:
			model.add(Dropout(rate=rate))
	model.add(Dense(1, activation='linear'))
	model.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', coeff_determination],optimizer=chosen_opt(learning_rate=learning_rate))
	input_units = X_train.shape[1]
	output_units = 1
	limit = math.sqrt(6 / (input_units + output_units)) #glorot_uniform limit
	beta_weights = np.array(beta_weights).reshape(-1, 1)
	beta_weights = np.interp(beta_weights, (beta_weights.min(), beta_weights.max()), (-(limit), limit)) #scale between two numbers

	#print(model.layers[0].get_weights()[0])
	#first_layer_weights = model.layers[0].get_weights()[0]
	#for i in first_layer_weights:
	#	print(max(i))
	#	print(min(i))
	#print(min(model.layers[0].get_weights()[0]))

	new_layer_weights = np.random.rand(X_train.shape[1],units) #(num_inputs,num_units)
	for i in range(0,X_train.shape[1]-1):
		new_layer_weights[i,:] = beta_weights[i]
	new_weight_list = []
	new_weight_list.append(new_layer_weights)
	new_weight_list.append(np.zeros(units)) # biases
	model.layers[0].set_weights(new_weight_list)
	new_layer_weights = model.layers[0].get_weights()[0]
	#for i in range(1,20):
	#	print(max(new_layer_weights[i]))
	#	print(min(new_layer_weights[i]))

	model.fit(X_train, y_train.ravel(), epochs=epochs, batch_size=batch_size, verbose=0 , validation_data=(X_test, y_test.ravel()))
	score = model.evaluate(X_test, y_test.ravel())[3] #should be R2
	f = open('seeded_score_file.txt', 'a') 
	f.write(str(score) + '\n')
	f.close()
	del(model)

	#print(model.layers[0].get_weights()[0])
	#first_layer_weights = model.layers[0].get_weights()[0]
	#for i in range(1,20):
	#       print(max(first_layer_weights[i]))
	#       print(min(first_layer_weights[i]))

	opt = HP_OPTIMIZER; chosen_opt = getattr(tf.keras.optimizers,opt)
	reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
	model2 = Sequential()
	model2.add(Dense(units=units, activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer, input_shape=(X_train.shape[1],)))
	if rate != 0:
		model2.add(Dropout(rate=rate))
	for i in range(HP_NUM_HIDDEN_LAYERS-1):
		model2.add(Dense(units=units, activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
		if rate != 0:
			model2.add(Dropout(rate=rate))
	model2.add(Dense(1, activation='linear'))
	model2.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', coeff_determination],optimizer=chosen_opt(learning_rate=learning_rate))
	model2.fit(X_train, y_train.ravel(), epochs=epochs, batch_size=batch_size, verbose=0 , validation_data=(X_test, y_test.ravel()))
	score = model2.evaluate(X_test, y_test.ravel())[3] #should be R2
	f = open('glorot_uni_score_file.txt', 'a') 
	f.write(str(score) + '\n')
	f.close()
	del(model2)
	count -= 1
	print(count)
