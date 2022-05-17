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

print("Please remember to set the right set size in the nested_cv code")
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

if organism != 'mouse':
	sys.path.insert(1, '/external_storage/ciaran/Library/Python/3.7/python/site-packages/nested_cv')
	import nested_cv
	from nested_cv import NestedCV
else:
	sys.path.insert(0, '/home/hers_en/rmclaughlin/tf/lib/python3.6/site-packages') ; sys.path.insert(0, '/hpc/local/CentOS7/modulefiles/python_libs/3.6.1'); sys.path.insert(0, '/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/envciaran2/lib/python3.6/site-packages')
	from nested_cv2 import NestedCV

#from sklearn.model_selection import cross_val_score
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

for i in range(1,len(sys.argv)):
	print(sys.argv[i])

if organism != 'mouse':
	if not os.path.exists('/external_storage/ciaran/' + organism + '/' + phenotype+ '/' + snps):
		os.makedirs('/external_storage/ciaran/' + organism + '/' + phenotype+ '/' + snps)

	os.chdir('/external_storage/ciaran/' + organism + '/' + phenotype+ '/' + snps)
else:
	if not os.path.exists('/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/' + organism + '/' + phenotype):
		os.makedirs('/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/' + organism + '/' + phenotype)
	os.chdir('/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/' + organism + '/'  + phenotype)


date_object = datetime.datetime.now().replace(second=0,microsecond=0)
print(date_object)

def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
        return (1-SS_res/SS_tot)


def load_data(data):
        dataset = np.loadtxt(data, skiprows=1, dtype='str')
        x = dataset[: , 6:set_size+6].astype(np.int) if organism != 'Arabadopsis' else dataset[: , 6:set_size+6]/2 #Arabdopsis data is inbred to homozyotisity to be 2/0
        y = dataset[: , 5 ].astype(np.float)
        y = y.reshape(-1,1)
        #print("Performing split of raw data....")
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
        return x, y #x_train, y_train, x_test, y_test



def baseline(x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model


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

def CK_nested_cv(x_outer_train, y_outer_train, x_outer_test, y_outer_test, estimator, param_grid, model_name):
        for key in param_grid:
                param_grid[key] = sorted(param_grid[key]) #need to sort for plotting
        kf_inner = sklearn.model_selection.KFold(n_splits=4, shuffle=True, random_state=42)
        kf_inner.get_n_splits(x_outer_train) #split outer train set
        model = BayesSearchCV(estimator=estimator, search_spaces=param_grid, n_jobs=32, n_points=1, n_iter=iterations, cv=kf_inner, refit=True, random_state=42, scoring=metric_in_use) #verbose=2
        result = model.fit(x_outer_train, y_outer_train)
        print(result.best_index_)
        print("Best inner score for fold %s is %s" % (count, result.best_score_))
        best_params = result.best_params_
        print("Best inner params for outer fold %s is %s" % (count, best_params))
        outer_score = model.score(x_outer_test, y_outer_test)
        _ = plot_objective(model.optimizer_results_[0],dimensions=list(best_params), n_minimum_search=int(1e8)) #partial dependance plots #will fail if under 10 iterations ("list index out of range")
        plt.show()
        plt.savefig("%s_cv_%s.png" % (model_name, count))
        plt.clf() ; plt.close()
        return outer_score

def loop_through(estimator, param_grid): #should be able to merge this with CK_nested_cv
        outer_scores = []
        count = 1 #for CK_nested_cv()
        metric_in_use = 'r2'
        outer_ks = 4 #number of outer splits
        model_name = ''.join(filter(str.isalnum, str(estimator).lower())) #grabs model name as string and removes ()
        for k in range(1, outer_ks+1):
                x_outer_train, y_outer_train = load_data('train_raw_plink_' + 'snps' + '_' + k + '_in_4_out.raw') #not standardized #already split
                x_outer_test, y_outer_test = load_data('test_raw_plink_' + 'snps' + '_' + k + '_in_4_out.raw') #not standardized #already split
		scaler = preprocessing.StandardScaler().fit(y_outer_train) 
		y_outer_train = scaler.transform(y_outer_train) ; y_outer_test = scaler.transform(y_outer_test)
                outer_score = CK_nested_cv(x_outer_train, y_outer_train, x_outer_test, y_outer_test, estimator=estimator, param_grid=param_grid, model_name=model_name) #e.g SVR()
                outer_scores.append(outer_score)
                count += 1
        print(outer_scores)
        print("Outer scores of estimator is %s and mean is %s" % (outer_scores, np.mean(outer_scores)))


#pickle.dump(scaler, open('scaler.pkl', 'wb'))
#scaler = pickle.load(open('scaler.pkl', 'rb'))

n_snps = x_train.shape[1]
metric_in_use = sklearn.metrics.r2_score if binary == 'False' else sklearn.metrics.roc_auc_score
#################################################SVM####SVM#####SVM####################################################################

print("Performing SVM")
c_param = [2e-2,2e-4,2e-8, 1,int(2e+2),int(2e+4),int(2e+8)] #can be negative #We found that trying exponentially growing sequences of C and γ is a practical method to identify good parameters https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
gamma_param = [0.002,0.2,0.5,0.01] #ValueError: gamma < 0
epsilon_param = [2e-5,2e-3,1,0]
loss_param = ['epsilon_insensitive', 'squared_epsilon_insensitive']
kernel_param = ['rbf', 'sigmoid'] #precompuited leads to square matrix error #temorarily removing poly for time reasons need to put it back in
tolerance=[1e-3,1e-5,1e-1]
shrinking=[True,False]
cache_size=[100,200,400]#Specify the size of the kernel cache (in MB).
degree = [1,2,3,0.1,100]
svm_random_grid = {'gamma':gamma_param, 'C':c_param,'kernel':kernel_param, "degree":degree, 'epsilon':epsilon_param, "shrinking":shrinking,"tol":tolerance,"cache_size":cache_size}
print(svm_random_grid)
svm_random_grid2 = {'C' : c_param, 'loss':loss_param, 'epsilon':epsilon_param}
print(svm_random_grid2)
rbg_goal_dict, rbg_time_dict = make_goal_dict(svm_random_grid)
svm_goal_dict, svm_time_dict = make_goal_dict(svm_random_grid2)
if binary == 'True':
	loss_param = ['squared_hinge']
	penalty_box = ['l1','l2'] #The combination of penalty='l1' and loss='hinge' is not supported
	dual = [False]
	svm_random_grid2 = {'C' : c_param, 'loss':loss_param, 'penalty':penalty_box, 'dual':dual}
	svm_goal_dict, svm_time_dict = make_goal_dict(svm_random_grid2)
	SVM_NCV = NestedCV(model_name='LinearSVC', name_list=name_list, num=num, model=LinearSVC(), goal_dict=svm_goal_dict, time_dict=svm_time_dict, params_grid=svm_random_grid2, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
else:
	loop_through(LinearSVR(), svm_random_grid2)	


#kf_outer = sklearn.model_selection.KFold(n_splits=4, shuffle=True, random_state=42) #should result in the exact same split as was done in line 341 nested_cv_new_name.py
#kf_outer.get_n_splits(X)


if binary == 'False' :
	print("Performing RBG")
	RBG_NCV = NestedCV(model_name='RBG', name_list=name_list, num=num, model=SVR(),  goal_dict=rbg_goal_dict, time_dict=rbg_time_dict,params_grid=svm_random_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
	RBG_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='RBG', goal_dict=rbg_goal_dict, time_dict=rbg_time_dict)
	ncv_results('RBG', RBG_NCV)

print("Performing LASSO")
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, -1, -10, -100]
max_iter=[1000,3000]
ridge_alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, -1, -10, -100]
tolerance=[1e-3,1e-5,1e-1]
selection=['cyclic','random']# default=’cyclic’
alpha_dict = {'alpha':alpha,"max_iter":max_iter, "tol":tolerance, "selection":selection}
ridge_alpha_dict = {'alpha':ridge_alpha, "tol":tolerance}
print(alpha_dict)
alpha_name_dict = {'alpha':"Alpha"}
lass_goal_dict, lass_time_dict = make_goal_dict(alpha_dict)
if binary == 'False' :
	LASS_NCV = NestedCV(model_name='LASS', name_list=name_list, num=num, model=Lasso(), goal_dict=lass_goal_dict, time_dict=lass_time_dict, params_grid=alpha_dict, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
	LASS_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='LASS', goal_dict=lass_goal_dict, time_dict=lass_time_dict)
	ncv_results('LASS', LASS_NCV)

print("Performing Ridge")
lass_goal_dict, lass_time_dict = make_goal_dict(ridge_alpha_dict)
if binary == 'True':
	RIDGE_NCV = NestedCV(model_name='RIDGE', name_list=name_list, num=num, model=RidgeClassifier(), goal_dict=lass_goal_dict, time_dict=lass_time_dict, params_grid=ridge_alpha_dict, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
else:
	RIDGE_NCV = NestedCV(model_name='RIDGE', name_list=name_list, num=num, model=Ridge(), goal_dict=lass_goal_dict, time_dict=lass_time_dict, params_grid=ridge_alpha_dict, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
RIDGE_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='RIDGE', goal_dict=lass_goal_dict, time_dict=lass_time_dict)
ncv_results('RIDGE', RIDGE_NCV)

print("Performing Random Forests")
n_estimators = [10,100,1000] # Number of features to consider at every split
max_features = ['sqrt', 'log2'] # Maximum number of levels in tree
max_depth = [1, 10, 50,100]
max_depth.append(None) # Minimum number of samples required to split a node
#min_samples_split = [int(x) for x in np.linspace(2, 2000, num = 100)]; min_samples_split.extend((5,10,20))
min_samples_split = [2, 10, 100, 1000] # Minimum number of samples required at each leaf node
#min_samples_leaf = [int(x) for x in np.linspace(1, 2000, num = 200)] ; min_samples_leaf.extend((2,4,8,16, 32, 64)) # Method of selecting samples for training each tree
min_samples_leaf = [1,2, 10, 100, 1000]
bootstrap = [False, True]
max_leaf_nodes = [10, 100, 500] #; max_leaf_nodes.append(x_train.shape[0])
max_samples = [0.5, 0.9, 0.1, 0.01]
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
if binary == 'True':
	RF_NCV = NestedCV(model_name='RF', name_list=name_list, num=num, model=RandomForestClassifier(), goal_dict=rf_goal_dict, time_dict=rf_time_dict, params_grid=random_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
else:
	RF_NCV = NestedCV(model_name='RF', name_list=name_list, num=num, model=RandomForestRegressor(), goal_dict=rf_goal_dict, time_dict=rf_time_dict, params_grid=random_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
RF_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='RF', goal_dict=rf_goal_dict, time_dict=rf_time_dict)
ncv_results('RF', RF_NCV)
#base_grid = {"fit_intercept":["True"]}
print("Performing Baseline")
base_goal_dict = {}
base_time_dict = {}
model_type = LinearRegression() if binary == 'False' else LogisticRegression()
BASELINE_NCV = NestedCV(model_name='baseline', name_list=name_list, num=num , model=model_type,goal_dict=base_goal_dict, time_dict=base_time_dict, params_grid={}, outer_kfolds=4, inner_kfolds=4, n_jobs = 2,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
BASELINE_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='baseline',goal_dict=base_goal_dict, time_dict=base_time_dict)
ncv_results('baseline', BASELINE_NCV)
print("Performing Neural Network")
param_grid = {'network_shape':['brick', 'funnel','long_funnel'], 'epochs' : [50,100,200],'batch_size' : [16,32, 128],'learning_rate' : [0.01, 0.001, 0.0001, 0.00001],'HP_L1_REG' : [1e-5,1e-6,1e-4, 1e-2, 0.1, 1e-3],'HP_L2_REG' : [1e-8, 1e-3, 1e-1, float(0)], 'kernel_initializer' : ['glorot_uniform', 'glorot_normal', 'random_normal', 'random_uniform', 'he_uniform', 'he_normal'],'activation' : ['tanh', 'relu', 'elu'],'HP_NUM_HIDDEN_LAYERS' : [2,3,5],'units' : [200, 100,1000], 'rate' : [float(0), 0.1, 0.3],'HP_OPTIMIZER' : ['Ftrl', 'RMSprop', 'Adadelta', 'Adamax', 'Adam', 'Adagrad', 'SGD']}
nn_goal_dict, nn_time_dict = make_goal_dict(param_grid)
METRIC_ACCURACY = coeff_determination
dependencies = {'coeff_determination':coeff_determination}
custom_objects = {"coeff_determination":coeff_determination}
tf.config.threading.set_inter_op_parallelism_threads(32)
tf.config.threading.set_intra_op_parallelism_threads(32)
#tf.config.experimental_run_functions_eagerly(True) #needed to avoid error # tensorflow.python.eager.core._SymbolicException

callback = tf.keras.callbacks.EarlyStopping(monitor='coeff_determination', patience=20, mode='max', baseline=0.0) #min above 0 #this callkaci is throwing up and error Unknown metric function

if binary == 'True': #overwrite variables
	dependencies = {'auc':tf.keras.metrics.AUC}
	METRIC_ACCURACY = tf.keras.metrics.AUC
	callback = tf.keras.callbacks.EarlyStopping(monitor='auc', patience=20, mode='max', baseline=0.0) #min above 0


import tempfile
#https://github.com/tensorflow/tensorflow/issues/34697 #fixes an error that the early stopping callback throws up in the nested cv #something about the parralele fitting step needing everything to be pickle-able and the callback isnt 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

make_keras_picklable()

def build_nn(HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, units, activation, learning_rate, HP_L1_REG, HP_L2_REG, rate, kernel_initializer, network_shape):
	opt = HP_OPTIMIZER
	chosen_opt = getattr(tf.keras.optimizers,opt)
	reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
	long_funnel_count = 0 #keep widest shape for two layers
	model = Sequential()
	input_shape = (x_train.shape[1],) if snps == 'shuf' else (x_train.shape[1]-1,)
	model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, input_shape=input_shape))
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
	#	new_Layer_weights[i,:] = beta_weights[i]
	#new_weight_list = []
	#new_weight_list.append(new_layer_weights)
	#new_weight_list.append(np.zeros(num_units)) # biases
	#model.layers[0].set_weights(new_weight_list)
	print(model.summary())
	return model


if binary == 'True':
	nn_model = KerasClassifier(build_fn = build_nn, verbose=0, callbacks=[callback])
else:
	nn_model = KerasRegressor(build_fn = build_nn, verbose=0, callbacks=[callback])


NN_NCV = NestedCV(model_name='nn_model', name_list=name_list, num=num, model=nn_model, goal_dict=nn_goal_dict, time_dict=nn_time_dict, params_grid=param_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
NN_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='NN', goal_dict=nn_goal_dict, time_dict=nn_time_dict)
nn_results('NN', NN_NCV)
print("Performing a convulutional neural network")
cnn_param_grid = {'network_shape':['brick', 'funnel','long_funnel'], 'epochs':[100, 50],'batch_size' : [16,64,128], 'learning_rate' : [0.01, 0.0001, 0.001],'HP_L1_REG' : [0.001, 0.0001,0.00001,0],'HP_L2_REG' : [0, 0.001,0.00001],'kernel_initializer' : ['glorot_normal', 'glorot_uniform', 'he_uniform', 'random_normal', 'random_uniform', 'he_normal'],'activation' : ['tanh', 'relu', 'elu'],'HP_NUM_HIDDEN_LAYERS' : [2,3, 5],'units' : [100,200,1000], 'rate' : [float(0), 0.1, 0.5],'HP_OPTIMIZER' : ['SGD','Ftrl', 'RMSprop', 'Adadelta', 'Adamax', 'Adam', 'Adagrad'], 'filters':[1,5],'strides':[1,2,3],'pool':[1,2,3],'kernel':[1,2,3]}
cnn_goal_dict, cnn_time_dict = make_goal_dict(cnn_param_grid)
if binary == 'True':
	METRIC_ACCURACY = tf.metrics.AUC
else:
	METRIC_ACCURACY = 'coeff_determination'

#not sure if strides is relevant
print(x_train.shape)
#K.set_image_dim_ordering('th') #Negative dimension size caused by subtracting 2 from 1 for 'MaxPool - fixes error
K.set_image_data_format('channels_first') #prevents error					      
def conv_model(HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, units, activation, learning_rate, HP_L1_REG, HP_L2_REG, rate, kernel_initializer,strides,pool,filters,kernel, network_shape):
        opt = HP_OPTIMIZER
        if HP_NUM_HIDDEN_LAYERS == 1 :
                print("HP_NUM_HIDDEN_LAYERS is equal to 1; this could cause building problems")
        chosen_opt = getattr(tf.keras.optimizers,opt)
        reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
        long_funnel_count = 0 #keep widest shape for two layers
        input_shape = (x_train.shape[1],1) if snps == 'shuf' else (x_train.shape[1]-1,1)
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
        
					      
cnn_model = KerasRegressor(build_fn = conv_model,verbose=0, callbacks=[callback]) if binary == 'False' else KerasClassifier(build_fn = conv_model,verbose=0, callbacks=[callback])
CNN_NCV = NestedCV(model_name='CNN', name_list=name_list, num=num,model=cnn_model, goal_dict=cnn_goal_dict, time_dict=cnn_time_dict, params_grid=cnn_param_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
CNN_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='CNN', goal_dict=cnn_goal_dict, time_dict=cnn_time_dict)
nn_results('CNN', CNN_NCV)
