#Warning : best model selected by NMAE and R2 might not be the same
#performs linear regression, linear regression, neural network, svm and random forest, LASSO, RIDGE, CNN
#source ~/venv/bin/activate #in python 3.5.2
#print a log to a .txt file!
#model = pickle.load(open('FILEPATH', 'rb')) 
#dependencies = {'coeff_determination':coeff_determination}
#model = tf.keras.models.load_model('FILEPATH', custom_objects=dependencies)
#

#import tensorflow
#import numpy as np; import scipy #need to do this before path insert
#import sys
#sys.path.insert(1, '/external_storage/ciaran/Library/Python/3.7/python/site-packages/')
#import dill as pickle
#sys.path.insert(1, '/external_storage/ciaran/Library/Python/3.7/python/site-packages/nested_cv')
#from nested_cv import NestedCV
#with open('NCV_NN.pkl', 'rb') as f:
#     red = pickle.load(f)

print("Please remember to set the right set size in the nested_cv code")
import sys
sys.path.insert(1, '/external_storage/ciaran/Library/Python/3.7/python/site-packages/nested_cv')
num = sys.argv[1] #script number for saving out
phenfile = str(sys.argv[2]) #txt file with phenotypes
data = str(sys.argv[3]) #needs to be same size as set_size
snps = str(sys.argv[4]) #top or shuf
phenotype = str(sys.argv[5]) #make a directory for the results
set_size = int(sys.argv[6]) #how many SNPs
from nested_cv import NestedCV
import statistics
import numpy as np
import sklearn
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
import time
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
#import pickle #use dill instead below
from sklearn.ensemble import RandomForestRegressor #based in part on https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
from statistics import mean
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import make_scorer
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten #.core
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from sklearn.externals import joblib
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # or Classifier
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
import random
#https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%202.ipynb
from tensorboard.plugins.hparams import api as hp
import random
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils



#if snps == 'shuf' :
#	print("Shuf nestedCV in usage")
#	from nested_cv_shuf import NestedCV
#elif snps == 'top':
#	from nested_cv import NestedCV
#else:
#	print("snnps must be top or shuf")
	

sys.path.insert(1, '/external_storage/ciaran/Library/Python/3.7/python/site-packages/')
import dill as pickle
for i in range(1,len(sys.argv)):
	print(sys.argv[i])

if not os.path.exists('/external_storage/ciaran/arabadopsis/' + phenotype+ '/' + snps):
    os.makedirs('/external_storage/ciaran/arabadopsis/' + phenotype+ '/' + snps)

os.chdir('/external_storage/ciaran/arabadopsis/' + phenotype+ '/' + snps)
date_object = datetime.datetime.now().replace(second=0,microsecond=0)
print(date_object)

def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
        return (1-SS_res/SS_tot)


def load_data(data):
        dataset = np.loadtxt(data, skiprows=1)
        x = dataset[: , 6:set_size+6]/2
        y = dataset[: , 5 ]
        y = y.reshape(-1,1)
        #print("Performing split of raw data....")
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
        return x, y #x_train, y_train, x_test, y_test



with open(('validation_results_'+ str(snps) +str(num) + phenotype + str("{:%Y_%m_%d}".format(datetime.datetime.now())) + '.vallog' ), 'a') as f:
        original_stdout = sys.stdout # Save a reference to the original standard output
        sys.stdout = f # Change the standard output to the file we created.
        print(datetime.datetime.now())
        sys.stdout = original_stdout      

def baseline(x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model


def avg_cv_result(measure,cv_result):
	my_var_name = [k for k,v in locals().items() if v == measure][0] #just to print out the name
	print(my_var_name)
	n_combos = len(cv_result.cv_results_['split0_test_neg_mean_absolute_error'])
	named_dict = {} #dictonary will have a list of results PER grid combination across all CV results and this will return the average result. 
	avg_list = []
	for combo in range(0, n_combos):
		named_dict[str(combo)] = []
		for split in measure:
			named_dict[str(combo)].append(cv_result.cv_results_[split][combo])
		avg_list.append(statistics.mean(named_dict[str(combo)]))
		print(combo, statistics.mean(named_dict[str(combo)]))

	print('Max', np.nanmax(avg_list), np.where(avg_list == np.nanmax(avg_list)))
	print('Min', np.nanmin(avg_list), np.where(avg_list == np.nanmin(avg_list)))
	return avg_list	


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

import collections
import operator

def make_param_box_plot(goal_dict, time_dict, analysis): #example goal dict = {'alpha' : {0.1 : [0.3, 0.5, 0.4], 1 : [0, 0.1, 0.2]}, 'beta' : {0.1 : [0.5, 0.5, 0.45, 1 : [0.8, 0.7, 0.7]}}
	if 'max_depth' in goal_dict.keys():
		if None in goal_dict['max_depth'].keys():
			old_key = None #thros up sorting error
			new_key = 0
			goal_dict['max_depth'][new_key] = goal_dict['max_depth'].pop(old_key)
			time_dict['max_depth'][new_key] = time_dict['max_depth'].pop(old_key)
	for param in goal_dict:
		for value in goal_dict[param]:
			goal_dict[param][value] = [0 if score < 0 else score for score in goal_dict[param][value]] #convert negative r2 to zeros
	for param in goal_dict:
                #ordered_dict_items = {k:goal_dict[param][k] for k in sorted(goal_dict[param].keys())} this doesnt work in python3.5 for some reason (does work in 3.8)
                #ordered_time_items = {k:time_dict[param][k] for k in sorted(time_dict[param].keys())}
                sorted_dict_items = sorted(goal_dict[param].items(), key=operator.itemgetter(0))#in order python It is not possible to sort a dictionary, only to get a representation of a dictionary that is sorted
                sorted_time_items = sorted(time_dict[param].items(), key=operator.itemgetter(0))
                ordered_dict_items = collections.OrderedDict(sorted_dict_items) #turn back into dictionary
                ordered_time_items = collections.OrderedDict(sorted_time_items)
                plt.subplots(1,2,figsize=(12,8))
                plt.subplot(121) #sorted
                plt.boxplot(ordered_dict_items.values(), bootstrap=None,showmeans=False, meanline=False, notch=True,labels=ordered_dict_items.keys()) #orange line is median, green dotted line is mean
                plt.xlabel(str(param).upper(), fontsize=10, fontweight='bold')
                plt.ylabel('R^2', fontsize=10,fontweight='bold')
                plt.title('R^2 Score vs %s' % param, fontsize=14, fontweight='bold')
                if param == 'initialization':
                        plt.xticks(fontsize=6)
                plt.subplot(122)
                plt.boxplot(ordered_time_items.values(), bootstrap=None,showmeans=False, meanline=False, notch=False,labels=ordered_time_items.keys())
                plt.xlabel(str(param).upper(), fontsize=10, fontweight='bold')
                plt.ylabel('Training Time', fontsize=10,fontweight='bold')
                plt.title('Training Time vs %s' % param, fontsize=14, fontweight='bold')
                plt.tight_layout(pad=4)
                if param == 'initialization':
                        plt.xticks(fontsize=6)
                my_fig_name = "plots_of_" +str(analysis) + '_' + str(param) + '_' + str("{:%Y_%m_%d}".format(datetime.datetime.now())) + '_' +str(snps) +str(num)+ ".png"
                plt.savefig(my_fig_name, dpi=300) 
                plt.show()
                plt.clf()
                plt.close()
		
def make_goal_dict(whole_dict):
	print(whole_dict)
	goal_dict = {key:{} for key in whole_dict}
	for key in whole_dict:
		for item in whole_dict[key]:
			goal_dict[key][item] = []
	time_dict = {key:{} for key in whole_dict} #both empty
	for key in whole_dict:
                for item in whole_dict[key]:
                        time_dict[key][item] = []
	return goal_dict, time_dict

x_train, y_train = load_data(data)
name_list = np.loadtxt(data, skiprows=1, usecols=(0,), dtype='str')

scaler = preprocessing.StandardScaler().fit(y_train)
#pickle.dump(scaler, open('scaler.pkl', 'wb'))
#scaler = pickle.load(open('scaler.pkl', 'rb'))

y_train = scaler.transform(y_train)

n_snps = x_train.shape[1]
my_cv = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
#################################################SVM####SVM#####SVM####################################################################
def ncv_results(analysis, ncv_object):
        print("Best Params of %s is %s " % (analysis, ncv_object.best_params))
        print("Outer scores of %s is %s and mean is %s" % (analysis, ncv_object.outer_scores, np.mean(ncv_object.outer_scores)))
        print("Variance of %s is %s " % (analysis, ncv_object.variance))
        #print("Goal dict of %s is %s " % (analysis, ncv_object.goal_dict))
        make_param_box_plot(ncv_object.goal_dict, ncv_object.time_dict, str(analysis))
        with open('NCV_' + str(analysis) + '_' +  str(snps) + '_' + str(phenotype) + '_' + str(num) + '.pkl', 'wb') as ncvfile: #with open("fname.pkl", 'rb') as ncvfile:
                pickle.dump(ncv_object, ncvfile) #ncv_object = pickle.load(ncvfile)

def nn_results(analysis, ncv_object):
        print("Best Params of %s is %s " % (analysis, ncv_object.best_params))
        print("Outer scores of %s is %s and mean is %s" % (analysis, ncv_object.outer_scores, np.mean(ncv_object.outer_scores)))
        print("Variance of %s is %s " % (analysis, ncv_object.variance))
        #print("Goal dict of %s is %s " % (analysis, ncv_object.goal_dict))
        make_param_box_plot(ncv_object.goal_dict, ncv_object.time_dict, str(analysis))
        nn_list = [ncv_object.best_inner_params_list, ncv_object.best_inner_score_list, ncv_object.best_params, ncv_object.metric, ncv_object.outer_scores, ncv_object.variance]
        with open('NCV_' + str(analysis) + '_' +  str(snps) + '_' + str(phenotype) + '_' + str(num) + '.pkl', 'wb') as ncvfile:
                pickle.dump(nn_list, ncvfile) #ncv_object = pickle.load(ncvfile)
        ncv_object.model.model.save("model_" + str(analysis) + '_' +  str(snps) + '_' + str(phenotype) + '_' + str(num) + ".h5")

'''
print("Performing SVM")
c_param = [2e-4,2e-3,2e-1,2e-2] #We found that trying exponentially growing sequences of C and ?? is a practical method to identify good parameters https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
gamma_param = [2e-4,2e-3,2e-1,2e-2]

epsilon_param = [2e-5,2e-3,1,int(2e+3),int(2e+5),0]
loss_param = ['epsilon_insensitive']
kernel_param = ['poly']
degree = [1,2,3]
svm_random_grid = {'gamma':gamma_param, 'C':c_param,'kernel':kernel_param, "degree":degree, 'epsilon':epsilon_param}
print(svm_random_grid)
svm_random_grid2 = {'C' : c_param, 'loss':loss_param, 'epsilon':epsilon_param}
print(svm_random_grid2)
rbg_goal_dict, rbg_time_dict = make_goal_dict(svm_random_grid)
svm_goal_dict, svm_time_dict = make_goal_dict(svm_random_grid2)
SVM_NCV = NestedCV(model_name='LinearSVR', name_list = name_list, model=LinearSVR(), goal_dict=svm_goal_dict, time_dict=svm_time_dict, params_grid=svm_random_grid2, outer_kfolds=4, inner_kfolds=4, n_jobs = 8,cv_options={'randomized_search':True, 'randomized_search_iter':50, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
SVM_NCV.fit(x_train, y_train.ravel(), name_list=name_list, phenfile=phenfile, set_size=set_size, snps=snps, model_name='SVM', goal_dict=svm_goal_dict, time_dict=svm_time_dict)
ncv_results('SVM', SVM_NCV)	
print("Performing RBG")
RBG_NCV = NestedCV(model_name='RBG', name_list=name_list, model=SVR(),  goal_dict=rbg_goal_dict, time_dict=rbg_time_dict,params_grid=svm_random_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 8,cv_options={'randomized_search':True, 'randomized_search_iter':50, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
RBG_NCV.fit(x_train, y_train.ravel(), name_list=name_list, phenfile=phenfile, set_size=set_size, snps=snps, model_name='RBG', goal_dict=rbg_goal_dict, time_dict=rbg_time_dict)
ncv_results('RBG', RBG_NCV)

print("Performing LASSO")
alpha = [0.05, 0.01, 0.005,50, 100, 200]
alpha_dict = {'alpha':alpha}
print(alpha_dict)
alpha_name_dict = {'alpha':"Alpha"}
lass_goal_dict, lass_time_dict = make_goal_dict(alpha_dict)
LASS_NCV = NestedCV(model_name='LASS', name_list=name_list, model=Lasso(), goal_dict=lass_goal_dict, time_dict=lass_time_dict, params_grid=alpha_dict, outer_kfolds=4, inner_kfolds=4, n_jobs = 8,cv_options={'randomized_search':True, 'randomized_search_iter':50, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
LASS_NCV.fit(x_train, y_train.ravel(), name_list=name_list, phenfile=phenfile, set_size=set_size, snps=snps, model_name='LASS', goal_dict=lass_goal_dict, time_dict=lass_time_dict)
ncv_results('LASS', LASS_NCV)
print("Performing Ridge")
lass_goal_dict, lass_time_dict = make_goal_dict(alpha_dict)
RIDGE_NCV = NestedCV(model_name='RIDGE', name_list=name_list, model=Ridge(), goal_dict=lass_goal_dict, time_dict=lass_time_dict, params_grid=alpha_dict, outer_kfolds=4, inner_kfolds=4, n_jobs = 8,cv_options={'randomized_search':True, 'randomized_search_iter':50, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
RIDGE_NCV.fit(x_train, y_train.ravel(), name_list=name_list, phenfile=phenfile, set_size=set_size, snps=snps, model_name='RIDGE', goal_dict=lass_goal_dict, time_dict=lass_time_dict)
ncv_results('RIDGE', RIDGE_NCV)

print("Performing Random Forests")
n_estimators = [10,100,1000] # Number of features to consider at every split
max_features = ['sqrt'] # Maximum number of levels in tree
max_depth = [10, 50, 100]
max_depth.append(None) # Minimum number of samples required to split a node
#min_samples_split = [int(x) for x in np.linspace(2, 2000, num = 100)]; min_samples_split.extend((5,10,20))
min_samples_split = [2,3,4, 10] # Minimum number of samples required at each leaf node
#min_samples_leaf = [int(x) for x in np.linspace(1, 2000, num = 200)] ; min_samples_leaf.extend((2,4,8,16, 32, 64)) # Method of selecting samples for training each tree
min_samples_leaf = [1,2,3, 10]
bootstrap = [False]
max_leaf_nodes = [100, 500, 1000] ; max_leaf_nodes.append(x_train.shape[0])
max_samples = [0.5, 0.9, 0.95, 0.99]
#{'max_depth': 46, 'max_leaf_nodes': 695, 'n_estimators': 2778, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'min_samples_split': 2, 'bootstrap': False, 'max_samples': 0.5}
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap, 'max_samples':max_samples, 'max_leaf_nodes':max_leaf_nodes}
print(random_grid)
rf_name_dict = {"max_samples":"Maximum Fraction of Samples", "max_leaf_nodes":"Maximum Leaf Nodes", "n_estimators":"Number of Estimators", "n_snps":"Number of SNPs","max_features":"Maximum Number of Features", "max_depth":"Maximum Depth", "min_samples_split":"Minimum Number of Samples to Split", "min_samples_leaf":"Minimum Number of Samples in Leaf"}
rf_param_dict = {'n_snps':'n_features', 'n_estimators':'n_estimators'}
rf_param_list = ['n_estimators','max_features','max_depth','min_samples_split','min_samples_leaf','max_leaf_nodes', 'max_samples'] #dont have bootstrap here
rf_goal_dict, rf_time_dict = make_goal_dict(random_grid)
RF_NCV = NestedCV(model_name='RF', name_list=name_list, model=RandomForestRegressor(), goal_dict=rf_goal_dict, time_dict=rf_time_dict, params_grid=random_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 8,cv_options={'randomized_search':True, 'randomized_search_iter':50, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
RF_NCV.fit(x_train, y_train.ravel(), name_list=name_list, phenfile=phenfile, set_size=set_size, snps=snps, model_name='RF', goal_dict=rf_goal_dict, time_dict=rf_time_dict)
ncv_results('RF', RF_NCV)
#base_grid = {"fit_intercept":["True"]}
print("Performing Baseline")
base_goal_dict = {}
base_time_dict = {}
BASELINE_NCV = NestedCV(model_name='baseline', name_list=name_list , model=LinearRegression(),goal_dict=base_goal_dict, time_dict=base_time_dict, params_grid={}, outer_kfolds=4, inner_kfolds=4, n_jobs = 2,cv_options={'randomized_search':True, 'randomized_search_iter':50, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
BASELINE_NCV.fit(x_train, y_train.ravel(), name_list=name_list, phenfile=phenfile, set_size=set_size, snps=snps, model_name='baseline',goal_dict=base_goal_dict, time_dict=base_time_dict)
ncv_results('baseline', BASELINE_NCV)
'''
import random
print("Performing Neural Network")
param_grid = {'epochs' : [50,100],'batch_size' : [8, 16,32],'learning_rate' : [0.01, 0.001,0.05],'HP_L1_REG' : [1e-4, 1e-3,1e-5],'HP_L2_REG' : [1e-8, 0.2, 1e-4, 1e-2], 'kernel_initializer' : ['glorot_uniform', 'glorot_normal'],'activation' : ['tanh', 'relu'],'HP_NUM_HIDDEN_LAYERS' : [2,3],'units' : [100,200], 'rate' : [float(0), 0.1, 0.2, 0.5],'HP_OPTIMIZER' : ['Adam', 'SGD', 'Adagrad']}
nn_goal_dict, nn_time_dict = make_goal_dict(param_grid)
METRIC_ACCURACY = coeff_determination
tf.config.threading.set_inter_op_parallelism_threads(64)
tf.config.threading.set_intra_op_parallelism_threads(64)
#tf.config.experimental_run_functions_eagerly(True) #needed to avoid error # tensorflow.python.eager.core._SymbolicException
callback = tf.keras.callbacks.EarlyStopping(monitor='coeff_determination', patience=20, mode='max', baseline=0.0) #min above 0
make_keras_picklable()
def build_nn(HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, units, activation, learning_rate, HP_L1_REG, HP_L2_REG, rate, kernel_initializer):
	opt = HP_OPTIMIZER
	chosen_opt = getattr(tf.keras.optimizers,opt)
	reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
	model = Sequential()
	model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer, input_shape=(x_train.shape[1],)))
	if rate != 0:
		model.add(Dropout(rate=rate))	
	for i in range(HP_NUM_HIDDEN_LAYERS-1):
		model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
		if rate != 0:
			model.add(Dropout(rate=rate))
	model.add(Dense(1, activation='linear'))
	model.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', coeff_determination],optimizer=chosen_opt(learning_rate=learning_rate))
	print(model.summary())
	return model


#regressor_keras = KerasRegressor(build_fn = build_nn, epochs=10, verbose=1, batch_size=32)
#pipeline_keras = Pipeline([('model', regressor_keras)])
nn_model = KerasRegressor(build_fn = build_nn, verbose=0, callbacks=[callback])

from sklearn.model_selection import cross_val_score


NN_NCV = NestedCV(model_name='nn_model', name_list = name_list, model=nn_model, goal_dict=nn_goal_dict, time_dict=nn_time_dict, params_grid=param_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'randomized_search':True, 'randomized_search_iter':20, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
NN_NCV.fit(x_train, y_train.ravel(), name_list=name_list, phenfile=phenfile, set_size=set_size, snps=snps, model_name='NN', goal_dict=nn_goal_dict, time_dict=nn_time_dict)
nn_results('NN', NN_NCV)
exit()
print("Performing a convulutional neural network")
from tensorboard.plugins.hparams import api as hp
import random
from tensorflow.keras.layers import Dense, Conv1D, Flatten


cnn_param_grid = {'epochs':[200, 100, 50],'batch_size' : [16,64], 'learning_rate' : [0.01,0.001, 0.0001],'HP_L1_REG' : [0.1, 0, 0.01],'HP_L2_REG' : [0.1, 0, 0.01],'kernel_initializer' : ['random_uniform', 'random_normal', 'glorot_normal', 'glorot_uniform'],'activation' : ['tanh', 'relu'],'HP_NUM_HIDDEN_LAYERS' : [2,3,4,5],'units' : [200,500], 'rate' : [float(0), 0.1, 0.5],'HP_OPTIMIZER' : ['SGD'], 'filters':[1,2,3],'strides':[1,2,3],'pool':[1,2],'kernel':[1,2,3]}
cnn_goal_dict, cnn_time_dict = make_goal_dict(cnn_param_grid)
METRIC_ACCURACY = 'coeff_determination'
#not sure if strides is relevant
print(x_train.shape)
#x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1]) # You needs to reshape your input data according to Conv1D layer input format - (batch_size, steps, input_dim)
#x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
#x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
#x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1]) # You needs to reshape your input data according to Conv1D layer input format - (batch_size, steps, input_dim)
print(x_train.shape)
#K.set_image_dim_ordering('th') #Negative dimension size caused by subtracting 2 from 1 for 'MaxPool - fixes error
K.set_image_data_format('channels_first')					      
def conv_model(HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, units, activation, learning_rate, HP_L1_REG, HP_L2_REG, rate, kernel_initializer,strides,pool,filters,kernel):
        opt = HP_OPTIMIZER
        if HP_NUM_HIDDEN_LAYERS == 1 :
                print("HP_NUM_HIDDEN_LAYERS is equal to 1; this could cause building problems")
        chosen_opt = getattr(tf.keras.optimizers,opt)
        reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
        model = Sequential() # Only use dropout on fully-connected layers, and implement batch normalization between convolutions.
        model.add(Conv1D(filters=filters, strides=strides, input_shape=(x_train.shape[1]-1,1),  padding='same',data_format='channels_last', activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer, kernel_size=kernel))
        model.add(tf.keras.layers.MaxPool1D(pool_size=pool, strides=strides,padding='same',data_format='channels_last'))
        for i in range(HP_NUM_HIDDEN_LAYERS-1):
                model.add(Conv1D(filters=filters, strides=strides, activation=activation,  padding='same',data_format='channels_last', kernel_regularizer=reg, kernel_initializer=kernel_initializer, kernel_size=kernel))
                model.add(tf.keras.layers.MaxPool1D(pool_size=pool, strides=strides,padding='same', data_format='channels_last'))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', coeff_determination],optimizer=chosen_opt(learning_rate=learning_rate))
        print("Summary ", model.summary())
        return model				      
        
					      
cnn_model = KerasRegressor(build_fn = conv_model,verbose=0, callbacks=[callback])
CNN_NCV = NestedCV(model_name='CNN', name_list=name_list,model=cnn_model, goal_dict=cnn_goal_dict, time_dict=cnn_time_dict, params_grid=cnn_param_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'randomized_search':True, 'randomized_search_iter':50, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
CNN_NCV.fit(x_train, y_train.ravel(), name_list=name_list, phenfile=phenfile, set_size=set_size, snps=snps, model_name='CNN', goal_dict=cnn_goal_dict, time_dict=cnn_time_dict)
nn_results('CNN', CNN_NCV)
