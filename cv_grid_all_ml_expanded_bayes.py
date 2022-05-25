#BAYESIAN NESTED CROSS_VALIDATION
#Warning : best model selected by NMAE and R2 might not be the same
#can be binary or continoous trait
#performs linear regression, logistic regression, neural network, svm and random forest, LASSO, RIDGE, CNN
#source ~/venv/bin/activate #in python 3.5.2
#if you need to download additonal packages you might need to load them into after expanding sys path like the dill importation below 
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

import sys
num = sys.argv[1] #script number for saving out
phenfile = str(sys.argv[2]) #txt file with phenotypes
data = str(sys.argv[3]) #needs to be same size as set_size
snps = str(sys.argv[4]) #top or shuf
phenotype = str(sys.argv[5]) #make a directory for the results
set_size = int(sys.argv[6]) #how many SNPs
organism = str(sys.argv[7]) #which directory mouse or arabadopsis (the mis-spelling is needed)
binary = str(sys.argv[8]) #True or False
binary_boolean = True if binary == 'True' else False
iterations = int(sys.argv[9])

if organism not in ['mouse', 'als_nest_top2'] :
	sys.path.insert(1, '/external_storage/ciaran/Library/Python/3.7/python/site-packages/nested_cv')
else:
	sys.path.insert(0, '/home/hers_en/rmclaughlin/tf/lib/python3.6/site-packages') ; sys.path.insert(0, '/hpc/local/CentOS7/modulefiles/python_libs/3.6.1'); sys.path.insert(0, '/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/envciaran2/lib/python3.6/site-packages')

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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # or Classifier
from sklearn.model_selection import KFold
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
import joblib
import tempfile
if binary == 'True':
	from sklearn.linear_model import LogisticRegression, RidgeClassifier
	from sklearn.ensemble import RandomForestClassifier
	from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
	from tensorflow.keras.metrics import AUC
	from sklearn.svm import LinearSVC
	from sklearn.svm import SVC


sys.path.insert(1, '/external_storage/ciaran/Library/Python/3.7/python/site-packages/')
import dill as pickle
from skopt import BayesSearchCV
from skopt.plots import plot_objective, plot_histogram
from skopt.space import Real, Categorical, Integer #https://scikit-optimize.github.io/stable/modules/space.html


def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
        return (1-SS_res/SS_tot)


def load_data(data):
        dataset = np.loadtxt(data, skiprows=1, dtype='str')
        x = dataset[: , 6:set_size+6].astype(np.int) if organism != 'Arabadopsis' else dataset[: , 6:set_size+6]/2 #Arabadopsis data is inbred to homozygotisity to be 2/0
        y = dataset[: , 5 ].astype(np.float)
        y = y.reshape(-1,1)
        #print("Performing split of raw data....")
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
        return x, y #x_train, y_train, x_test, y_test



def baseline(x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model


def unpack(model, training_config, weights): ##https://github.com/tensorflow/tensorflow/issues/34697 #
    """fixes an error that the early stopping callback throws up in the nested cv 
    Something about the parralele fitting step needing everything to be pickle-able and the callback isnt""" 
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


def CK_nested_cv(x_outer_train, y_outer_train, x_outer_test, y_outer_test, estimator, param_grid, model_name, k):
        """Fits each Cross-validation fold within an inner loop. 
            Applies best params to outer test set and reports results. 
            Generates HP performance plots."""
        kf_inner = sklearn.model_selection.KFold(n_splits=4, shuffle=True, random_state=42)
        kf_inner.get_n_splits(x_outer_train) #split outer train set
        print(kf_inner)
        if organism in ['mouse', 'als_nest_top2']:
                num_jobs = 1
        else:
                num_jobs = 32
        if model_name in ('FNN' , 'CNN'):
                model = BayesSearchCV(estimator=estimator, search_spaces=param_grid, fit_params={'verbose':0, 'callbacks': [callback1]}, n_jobs=1, n_points=12, n_iter=iterations, cv=kf_inner, refit=True, random_state=42, scoring=metric_in_use) #n_jobs > 1 for NNs leads to a parallelism error "A task has failed to un-serialize"
        else:
                model = BayesSearchCV(estimator=estimator, search_spaces=param_grid, n_jobs=num_jobs, n_points=1, n_iter=iterations, cv=kf_inner, refit=True, random_state=42, scoring=metric_in_use) #verbose=2 gives more info
        result = model.fit(x_outer_train, y_outer_train.ravel()) #raveling is reshaping
        print(result.best_index_)
        print("Best %s inner score for fold %s is %s" % (model_name ,k, result.best_score_))
        best_params = result.best_params_
        print("Best %s inner params for outer fold %s is %s" % (model_name, k, best_params))
        outer_score = model.score(x_outer_test, y_outer_test)
        print("Score for %s outer fold %s is %s" % (model_name, k,outer_score))
        print(best_params)
        scores = model.cv_results_['split0_test_score'] + model.cv_results_['split1_test_score'] + model.cv_results_['split2_test_score'] + model.cv_results_['split3_test_score'] #needs edit to change with k
        for param in model.cv_results_['params'][0].keys():
                list_name = param + '_list' #create string 
                globals()[list_name] = [] #convert string to a list with a variable name
                for i in range(0,iterations):
                        globals()[list_name].append(model.cv_results_['params'][i][param])
                globals()[list_name] = globals()[list_name] * kf_inner.n_splits
                plt.scatter(globals()[list_name], scores)
                plt.xlabel(str(param).upper(), fontsize=10, fontweight='bold')
                plt.ylabel(metric_in_use.upper(), fontsize=10,fontweight='bold')
                if param == 'initialization':
                        plt.xticks(fontsize=6)
                plt.title('%s Score vs %s' % (metric_in_use.upper(), param), fontsize=14, fontweight='bold')
                if hasattr(param_grid[param], 'prior'):
                        if param_grid[param].prior == 'log_uniform': #whether or not to use log-sclae for x-axis
                                plt.xscale('log')
                plt.show() ; plt.savefig("%s_%s_cv_%s_%s_%s.png" % (param, model_name, k, snps, num), dpi=300) ; plt.clf() ; plt.close()
        return outer_score

def loop_through(estimator, param_grid, model_name): #should be able to merge this with CK_nested_cv()
        """Loops through each outer fold of the nested cv to run CK_nested_cv()"""
        outer_scores = []
        outer_ks = 4 #number of outer splits
        for k in range(1, outer_ks+1):
                x_outer_train, y_outer_train = load_data('train_raw_plink_' + snps + '_' + str(k) + '_in_4_out.raw') #not standardized #already split
                x_outer_test, y_outer_test = load_data('test_raw_plink_' + snps + '_' + str(k) + '_in_4_out.raw') #not standardized #already split
                if binary == 'False' : 
                        scaler = preprocessing.StandardScaler().fit(y_outer_train) 
                        y_outer_train = scaler.transform(y_outer_train) ; y_outer_test = scaler.transform(y_outer_test)
                else:
                        y_outer_train = (y_outer_train -1) ; y_outer_test =  (y_outer_test -1)
                        #x_outer_train = (x_outer_train/2) ; x_outer_test = (x_outer_test/2)
                if model_name == 'CNN':
                        x_outer_train = x_outer_train.reshape(x_outer_train.shape[0],x_outer_train.shape[1],1)
                        x_outer_test = x_outer_test.reshape(x_outer_test.shape[0],x_outer_test.shape[1],1)
                outer_score = CK_nested_cv(x_outer_train, y_outer_train, x_outer_test, y_outer_test, estimator=estimator, param_grid=param_grid, model_name=model_name, k=k) #e.g SVR()
                outer_scores.append(outer_score)
        print(outer_scores)
        print("Outer scores of %s is %s and mean is %s" % (model_name, outer_scores, np.mean(outer_scores)))


for i in range(1,len(sys.argv)):
        print(sys.argv[i])

if organism not in ['mouse', 'als_nest_top2']:
        if not os.path.exists('/external_storage/ciaran/' + organism + '/' + phenotype+ '/' + snps):
                os.makedirs('/external_storage/ciaran/' + organism + '/' + phenotype+ '/' + snps)

        os.chdir('/external_storage/ciaran/' + organism + '/' + phenotype+ '/' + snps)
else:
        if not os.path.exists('/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/' + organism ):
                os.makedirs('/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/' + organism )
        os.chdir('/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/' + organism )


print("Warning: if you get this error: 'xi, yi = partial_dependence_1D(space, result.models[-1], ;  IndexError: list index out of range' then increase n_iteations to >= 10")
date_object = datetime.datetime.now().replace(second=0,microsecond=0)
print(date_object)

#pickle.dump(scaler, open('scaler.pkl', 'wb'))
#scaler = pickle.load(open('scaler.pkl', 'rb'))

#metric_in_use = sklearn.metrics.r2_score if binary == 'False' else sklearn.metrics.roc_auc_score
#################################################SVM####SVM#####SVM####################################################################

if binary == 'False':
	metric_in_use = 'r2' #'neg_mean_squared_error' 
elif binary == 'True':
	metric_in_use = 'roc_auc'

print("Metric in use is %s" % metric_in_use)
'''
print("Performing SVM")
c_param = Real(2e-2,int(2e+8), prior='log_uniform') #can be negative #We found that trying exponentially growing sequences of C and γ is a practical method to identify good parameters https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
rbf_c = Real(2e-2,int(2e+8), prior='log_uniform')
gamma_param = Real(0.0001,1, prior='log_uniform') #ValueError: gamma < 0
epsilon_param = Real(1e-6,1, prior='log_uniform')
loss_param = ['epsilon_insensitive', 'squared_epsilon_insensitive']
kernel_param = ['rbf', 'sigmoid'] #precompuited leads to square matrix error #temorarily removing poly for time reasons need to put it back in
tolerance= Real(1e-5,1, prior='log_uniform')
shrinking=[True,False]
cache_size= Integer(100,1000, prior='uniform')#Specify the size of the kernel cache (in MB).
degree = Real(0.1,100, prior='log_uniform')

svm_random_grid = {'gamma':gamma_param, 'C':rbf_c,'kernel':kernel_param, "degree":degree, 'epsilon':epsilon_param, "shrinking":shrinking,"tol":tolerance,"cache_size":cache_size}
print(svm_random_grid)
svm_random_grid2 = {'C' : c_param, 'loss':loss_param, 'epsilon':epsilon_param}
print(svm_random_grid2)
if binary == 'True':
	loss_param = ['squared_hinge']
	penalty_box = ['l1','l2'] #The combination of penalty='l1' and loss='hinge' is not supported
	dual = [False]
	svm_random_grid2 = {'C' : c_param, 'loss':loss_param, 'penalty':penalty_box, 'dual':dual}
	loop_through(LinearSVC(), svm_random_grid2, 'linSVC')
elif binary == 'False':
	loop_through(LinearSVR(), svm_random_grid2, 'linSVR')	

#kf_outer = sklearn.model_selection.KFold(n_splits=4, shuffle=True, random_state=42) #should result in the exact same split as was done in line 341 nested_cv_new_name.py
#kf_outer.get_n_splits(X)

if binary == 'False' :
	print("Performing RBG")
	loop_through(SVR(), svm_random_grid, 'SVR')


print("Performing LASSO")
alpha = Real(-100, 1000, prior='log_uniform')
max_iter=Integer(1000,3000, prior='uniform')
ridge_alpha = Real(-100, 1000, prior='log_uniform')
tolerance=Real(1e-5,1e-1, prior='uniform')
selection=['cyclic','random']# default=’cyclic’
alpha_dict = {'alpha':alpha,"max_iter":max_iter, "tol":tolerance, "selection":selection}
ridge_alpha_dict = {'alpha':ridge_alpha, "tol":tolerance}
print(alpha_dict)
alpha_name_dict = {'alpha':"Alpha"}

if binary == 'False' :
	loop_through(Lasso(), alpha_dict, 'LASSO')

print("Performing Ridge")
if binary == 'True':
	loop_through(RidgeClassifier(), ridge_alpha_dict, 'Ridge')
else:
	loop_through(Ridge(), ridge_alpha_dict, 'Ridge')

print("Performing Random Forests")
n_estimators = Integer(10,1000, prior='log_uniform') # Number of features to consider at every split
max_features = ['sqrt', 'log2'] # Maximum number of levels in tree
max_depth = Integer(1, 1000, prior='log_uniform')
#min_samples_split = [int(x) for x in np.linspace(2, 2000, num = 100)]; min_samples_split.extend((5,10,20))
min_samples_split = Integer(2,1000,prior='log_uniform') # Minimum number of samples required at each leaf node
#min_samples_leaf = [int(x) for x in np.linspace(1, 2000, num = 200)] ; min_samples_leaf.extend((2,4,8,16, 32, 64)) # Method of selecting samples for training each tree
min_samples_leaf = Integer(1,1000,prior='log_uniform')
bootstrap = [False, True]
max_leaf_nodes = Integer(10,500,prior='log_uniform') #; max_leaf_nodes.append(x_train.shape[0])
max_samples = Real(0.01, 0.99, prior='log_uniform')
#{'max_depth': 46, 'max_leaf_nodes': 695, 'n_estimators': 2778, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'min_samples_split': 2, 'bootstrap': False, 'max_samples': 0.5}
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap':bootstrap,
               'max_samples':max_samples, 'max_leaf_nodes':max_leaf_nodes}
print(random_grid)
rf_name_dict = {"max_samples":"Maximum Fraction of Samples", "max_leaf_nodes":"Maximum Leaf Nodes", "n_estimators":"Number of Estimators", "n_snps":"Number of SNPs","max_features":"Maximum Number of Features", "max_depth":"Maximum Depth", "min_samples_split":"Minimum Number of Samples to Split", "min_samples_leaf":"Minimum Number of Samples in Leaf"}
rf_param_dict = {'n_snps':'n_features', 'n_estimators':'n_estimators'}
rf_param_list = ['n_estimators','max_features','max_depth','min_samples_split','min_samples_leaf','max_leaf_nodes', 'max_samples'] #dont have bootstrap here
if binary == 'True':
	loop_through(RandomForestClassifier(), random_grid, 'RF')
else:
	loop_through(RandomForestRegressor(), random_grid, 'RF')

#base_grid = {"fit_intercept":["True"]}
#print("Performing Baseline")
#base_goal_dict = {}
#base_time_dict = {}
#model_type = LinearRegression() if binary == 'False' else LogisticRegression()
#BASELINE_NCV = NestedCV(model_name='baseline', name_list=name_list, num=num , model=model_type,goal_dict=base_goal_dict, time_dict=base_time_dict, params_grid={}, outer_kfolds=4, inner_kfolds=4, n_jobs = 2,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
#BASELINE_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='baseline',goal_dict=base_goal_dict, time_dict=base_time_dict)
#ncv_results('baseline', BASELINE_NCV)
'''
METRIC_ACCURACY = coeff_determination
dependencies = {'coeff_determination':coeff_determination}
custom_objects = {"coeff_determination":coeff_determination}
tf.config.threading.set_inter_op_parallelism_threads(64)
tf.config.threading.set_intra_op_parallelism_threads(64)

callback1 = tf.keras.callbacks.EarlyStopping(monitor='coeff_determination', patience=20, mode='max', baseline=0.0)
print("Performing Neural Network")
param_grid = {'network_shape':['brick', 'funnel','long_funnel'], 'epochs' : Integer(50,500, prior='uniform'),'batch_size' : Integer(16,128, prior='uniform'),'learning_rate' : Real(1e-7, 1e-2, prior='log_uniform'),'HP_L1_REG' : Real(1e-6,0.1, prior='log_uniform'),'HP_L2_REG' : Real(1e-8,1e-1 ,prior='log_uniform'), 'kernel_initializer' : ['glorot_uniform', 'glorot_normal', 'random_normal', 'random_uniform', 'he_uniform', 'he_normal'],'activation' : ['tanh', 'relu', 'elu'],'HP_NUM_HIDDEN_LAYERS' : Integer(2,5, prior='uniform'),'units' : Integer(100,1000, prior='uniform'), 'rate' : Real(float(0),0.5, prior='uniform'),'HP_OPTIMIZER' : ['Ftrl', 'RMSprop', 'Adadelta', 'Adamax', 'Adam', 'Adagrad', 'SGD']}
#tf.config.experimental_run_functions_eagerly(True) #needed to avoid error # tensorflow.python.eager.core._SymbolicException

if binary == 'True': #overwrite variables
	dependencies = {'auc':tf.keras.metrics.AUC}
	METRIC_ACCURACY = tf.keras.metrics.AUC
	callback1 = tf.keras.callbacks.EarlyStopping(monitor='auc', patience=20, mode='max', baseline=0.0) #min above 0


#https://github.com/tensorflow/tensorflow/issues/34697 #fixes an error that the early stopping callback throws up in the nested cv #something about the parralele fitting step needing everything to be pickle-able and the callback isnt 
make_keras_picklable()

def build_nn(HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, units, activation, learning_rate, HP_L1_REG, HP_L2_REG, rate, kernel_initializer, network_shape):
        opt = HP_OPTIMIZER
        chosen_opt = getattr(tf.keras.optimizers,opt)
        reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
        long_funnel_count = 0 #keep widest shape for two layer
        make_keras_picklable()
        model = Sequential()
        input_shape = (set_size,) #if snps == 'shuf' else (set_size-1,)
        model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, input_shape=input_shape, kernel_initializer=kernel_initializer))
        if rate != 0:
                model.add(Dropout(rate=rate))
        for i in range(HP_NUM_HIDDEN_LAYERS-1):
                if network_shape == 'funnel':
                        units = int(units*0.666)
                elif network_shape == 'long_funnel':
                        if long_funnel_count >= 1: #two wide layers (inclduing previous first layer)
                                units=int(units*0.666)
                        long_funnel_count += 1
                model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
                if rate != 0:
                        model.add(Dropout(rate=rate))
        if binary == 'True' :
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy',metrics=['accuracy', AUC(name='auc')],optimizer=chosen_opt(learning_rate=learning_rate))
        else:
                model.add(Dense(1, activation='linear'))
                model.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', coeff_determination],optimizer=chosen_opt(learning_rate=learning_rate))
        #new_layer_weights = np.random.rand(x_train.shape[1]-1,units) #(num_inputs,num_units)
        #for i in range(0,x_train.shape[1]-1):
        #       new_Layer_weights[i,:] = beta_weights[i]
        #new_weight_list = []
        #new_weight_list.append(new_layer_weights)
        #new_weight_list.append(np.zeros(num_units)) # biases
        #model.layers[0].set_weights(new_weight_list)
        print(model.summary())
        return model

if binary == 'True':
	nn_model = KerasClassifier(build_fn = build_nn) #put fit params like verbose and callbacks in the CK_nested_cv() function
	loop_through(nn_model, param_grid, 'FNN')
else:
	nn_model = KerasRegressor(build_fn = build_nn) #put fit params like verbose and callbacks in the CK_nested_cv() function
	loop_through(nn_model, param_grid, 'FNN')


exit()
print("Performing a convulutional neural network")
#can't have zeros in any of these params as it will throw up a "Not all points are within the bounds of the space." error
cnn_param_grid = {'network_shape':['brick', 'funnel','long_funnel'], 'epochs':Integer(50,500, prior='uniform'),'batch_size' : Integer(16,128, prior='uniform'), 'learning_rate' : Real(0.0001, 0.01,prior='log_uniform'),'HP_L1_REG' : Real(0.000001,0.001,prior='log_uniform'),'HP_L2_REG' : Real(0.000001, 0.001,prior='log_uniform'),'kernel_initializer' : ['glorot_normal', 'glorot_uniform', 'he_uniform', 'random_normal', 'random_uniform', 'he_normal'],'activation' : ['tanh', 'relu', 'elu'],'HP_NUM_HIDDEN_LAYERS' : Integer(2, 5,prior='uniform'),'units' : Integer(100,1000,prior='uniform'), 'rate' : Real(float(0),0.5,prior='uniform'),'HP_OPTIMIZER' : ['SGD','Ftrl', 'RMSprop', 'Adadelta', 'Adamax', 'Adam', 'Adagrad'], 'filters':Integer(1,10,prior='uniform'),'strides':Integer(1,10, prior='uniform'),'pool':Integer(1,10, prior='uniform'),'kernel':Integer(1,10, prior='uniform')}
if binary == 'True':
	METRIC_ACCURACY = tf.metrics.AUC
else:
	METRIC_ACCURACY = 'coeff_determination'

#not sure if strides is relevant
#K.set_image_dim_ordering('th') #Negative dimension size caused by subtracting 2 from 1 for 'MaxPool - fixes error
K.set_image_data_format('channels_first') #prevents error					      
def conv_model(HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, units, activation, learning_rate, HP_L1_REG, HP_L2_REG, rate, kernel_initializer,strides,pool,filters,kernel, network_shape):
        opt = HP_OPTIMIZER
        if HP_NUM_HIDDEN_LAYERS == 1 :
                print("HP_NUM_HIDDEN_LAYERS is equal to 1; this could cause building problems")
        chosen_opt = getattr(tf.keras.optimizers,opt)
        reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
        long_funnel_count = 0 #keep widest shape for two layers
        input_shape = (set_size,1) #if snps == 'shuf' else (set_size-1,1)
        model = Sequential() # Only use dropout on fully-connected layers, and implement batch normalization between convolutions.
        model.add(Conv1D(filters=filters, strides=strides, input_shape=input_shape,  padding='same',data_format='channels_last', activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer, kernel_size=kernel))
        model.add(tf.keras.layers.MaxPool1D(pool_size=pool, strides=strides,padding='same',data_format='channels_last'))
        for i in range(HP_NUM_HIDDEN_LAYERS-1):
                if network_shape == 'funnel':
                        units = int(units*0.666)
                elif network_shape == 'long_funnel':
                        if long_funnel_count >= 1: #two wide layers (inclduing previous first layer)
                                units=int(units*0.666)
                        long_funnel_count += 1
                model.add(Conv1D(filters=filters, strides=strides, activation=activation,  padding='same',data_format='channels_last', kernel_regularizer=reg, kernel_initializer=kernel_initializer, kernel_size=kernel))
                model.add(tf.keras.layers.MaxPool1D(pool_size=pool, strides=strides,padding='same', data_format='channels_last'))
        model.add(Flatten())
        if binary == 'True':
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy',metrics=['accuracy', AUC(name='auc')],optimizer=chosen_opt(learning_rate=learning_rate))
        else:
                model.add(Dense(1, activation='linear'))
                model.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', coeff_determination],optimizer=chosen_opt(learning_rate=learning_rate))
        print("Summary ", model.summary())
        return model				      
        
					      
if binary == 'True':
        cnn_model = KerasClassifier(build_fn = conv_model, verbose=0)#, callbacks=[callback])
        loop_through(cnn_model, cnn_param_grid, 'CNN')
else:
        cnn_model = KerasRegressor(build_fn = conv_model, verbose=0)# , callbacks=[callback])
        loop_through(cnn_model, cnn_param_grid, 'CNN')

