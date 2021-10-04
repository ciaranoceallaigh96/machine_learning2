#Warning : best model selected by NMAE and R2 might not be the same
#performs linear regression, linear regression, neural network, svm and random forest, LASSO, RIDGE, CNN
#source ~/venv/bin/activate #in python 3.5.2
#print a log to a .txt file!
#model = pickle.load(open('FILEPATH', 'rb')) 
#dependencies = {'coeff_determination':coeff_determination}
#model = tf.keras.models.load_model('FILEPATH', custom_objects=dependencies)
import sys
sys.path.insert(1, '/external_storage/ciaran/Library/Python/3.7/python/site-packages/nested_cv')
num = sys.argv[1] #script number for saving out
nahnah = str(sys.argv[2]) #nothing rn
data = str(sys.argv[3])
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
import pickle
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # or Classifier
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
import random
from tensorboard.plugins.hparams import api as hp
#https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%202.ipynb
from tensorboard.plugins.hparams import api as hp
import random
for i in range(1,len(sys.argv)):
	print(sys.argv[i])

if not os.path.exists('/home/ciaran/arabadopsis/' + phenotype):
    os.makedirs('/home/ciaran/arabadopsis/' + phenotype)

os.chdir('/home/ciaran/arabadopsis/' + phenotype)
date_object = datetime.datetime.now().replace(second=0,microsecond=0)
print(date_object)

def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
        return (1-SS_res/SS_tot)

def coeff_determination2(y_true, y_pred):
        y_true = tf.convert_to_tensor(value=y_true, dtype='float32'); y_pred = tf.convert_to_tensor(value=y_pred, dtype='float32')
        SS_res = K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
        return float(1-SS_res/SS_tot)

def plot_results(model, param, group, analysis, name_dict):
        param_name = 'param_%s' % param
        train_scores = model.cv_results_['mean_train_score']
        test_scores = model.cv_results_['mean_test_score']
        train_time = model.cv_results_['mean_fit_time']
        param_values = list(model.cv_results_[param_name])
        plt.subplots(1,2,figsize=(12,8))
        #
        plt.subplot(121)
        plt.scatter(param_values, train_scores, label="train") #plt.plot(param_values, train_scores, 'bo-', label="train")
        plt.scatter(param_values, test_scores, label="test") # m_values, test_scores, 'go-', label="test")
        min_lim = min(min(train_scores), min(test_scores))
        max_lim = max(max(train_scores), max(test_scores))
        max_lim += (np.abs(max_lim)*0.1)
        min_lim -= (np.abs(min_lim)*0.1)
        plt.ylim(ymin=min_lim,ymax=max_lim)
        plt.legend(bbox_to_anchor=(-0.6,-0.3), loc='lower left')
        plt.xlabel(name_dict[param], fontsize=10)
        plt.ylabel('Neg Mean Absolute Error', fontsize=10)
        xticks, xticklabels = plt.xticks() ;  yticks, yticklabels = plt.yticks() #needed to stop edge points being cut off
        # shift half a step to the left
        # x0 - (x1 - x0) / 2 = (3 * x0 - x1) / 2
        xmin = (3*xticks[0] - xticks[1])/2. ; ymin = (3*yticks[0] - yticks[1])/2.
        # shift half a step to the right
        xmax = (3*xticks[-1] - xticks[-2])/2. ; ymax = (3*yticks[-1] - yticks[-2])/2.
        plt.xlim(xmin, xmax) ; plt.ylim(ymin, ymax)
        plt.xticks(xticks) ; plt.yticks(yticks)
        plt.title('Score vs %s' % name_dict[param], fontsize=10, fontweight='bold')
        #
        plt.subplot(122)
        plt.scatter(param_values, train_time) #'ro-')
        max_time = max(train_time) * 1.1
        plt.ylim(ymin=-5,ymax=max_time)
        plt.xlabel(name_dict[param], fontsize=10)
        plt.ylabel("Training Time (s) ", fontsize=10)
        plt.title("Training Time vs %s " % name_dict[param], fontsize=10, fontweight='bold')
        xticks, xticklabels = plt.xticks() ;  yticks, yticklabels = plt.yticks() #needed to stop edge points being cut off
        # shift half a step to the left
        # x0 - (x1 - x0) / 2 = (3 * x0 - x1) / 2
        xmin = (3*xticks[0] - xticks[1])/2. ; ymin = (3*yticks[0] - yticks[1])/2.
        # shift half a step to the right
        xmax = (3*xticks[-1] - xticks[-2])/2. ; ymax = (3*yticks[-1] - yticks[-2])/2.
        plt.xlim(xmin, xmax) ; plt.ylim(ymin, ymax)
        plt.xticks(xticks) ; plt.yticks(yticks)
        #
        plt.tight_layout(pad=4)
        myfigname = "plots_of_" + str(analysis) + '_' + str(snps) + '_' + str(param) + '_' + str("{:%Y_%m_%d}".format(datetime.datetime.now())) + "_" +str(group) + '_' + str(num)+ ".png"
        plt.savefig(myfigname, dpi=300)#
        print("%s saved to file!" % myfigname)
        plt.clf()
        plt.close()

def plot_search_results(grid):
    #can be chnaged from r2 to neagtive_mean_absolute_error
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_r2']
    stds_test = results['std_test_r2']
    means_train = results['mean_train_r2']
    stds_train = results['std_train_r2']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        print(i, p)
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()
    plt.savefig('try_fig2', dpi=300)
    plt.clf()
    plt.close()

def load_data(data):
        dataset = np.loadtxt(data, skiprows=1)
        x = dataset[: , 6:set_size]/2
        y = dataset[: , 5 ]
        y = y.reshape(-1,1)
        #print("Performing split of raw data....")
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
        return x, y #x_train, y_train, x_test, y_test



def evaluation(model, x, y, group):
        score = model.score(x,y)
        predictions = model.predict(x)
        errors = abs(predictions - y)
        mape = 100* np.mean(errors / y)
        accuracy = 100 - mape
        print("Model Performance - %s" % group)
        #print("Average Error: .4f% degrees" % np.mean(errors))
        #print("Accuracy = " + str(accuracy) +"%")
        print("R2 = " + str(score) + "%")
        return accuracy

with open(('validation_results_'+ str(snps) +str(num) + phenotype + str("{:%Y_%m_%d}".format(datetime.datetime.now())) + '.vallog' ), 'a') as f:
        original_stdout = sys.stdout # Save a reference to the original standard output
        sys.stdout = f # Change the standard output to the file we created.
        print(datetime.datetime.now())
        sys.stdout = original_stdout      

def valuation(model, x, y, group):
        print("Print validation results to separate log file")
        print('validation_results_'+ str(snps) +str(num) + phenotype + str("{:%Y_%m_%d}".format(datetime.datetime.now())) + '.vallog')
        original_stdout = sys.stdout # Save a reference to the original standard output
        score = model.score(x,y)
        predictions = model.predict(x)
        errors = abs(predictions - y)
        mape = 100* np.mean(errors / y)
        accuracy = 100 - mape
        with open(('validation_results_'+ str(snps) +str(num) + phenotype + str("{:%Y_%m_%d}".format(datetime.datetime.now())) + '.vallog' ), 'a') as f:
                sys.stdout = f # Change the standard output to the file we created.
                print("Model Performance - %s" % group)
                print("Average Error: .4f% degrees" % np.mean(errors))
                print("Accuracy = " + str(accuracy) +"%")
                print("R2 = " + str(score) + "%")
                print("MSE = " + str(sklearn.metrics.mean_squared_error(y, predictions)))
        sys.stdout = original_stdout # Reset the standard output to its original value
        return accuracy

def eval_model(model, x_train, y_train, x_test, y_test):
        n_trees = model.get_params()['estimator__n_estimators']
        n_snps = x_train.shape[1]
        predictions = []
        run_times = []
        for i in range(10):
                start_time = time.time()
                model.fit(x_train, y_train.ravel())
                predictions.append(model.predict(x_test))
                end_time = time.time()
                run_times.append(end_time - start_time)
        run_time = np.mean(run_times)
        predictions = np.mean(np.array(predictions), axis=0)
        errors = abs(predictions - y_test)
        mean_error = np.mean(errors)
        mape = 100 * np.mean(errors / y_test) #Mean absolute percentage error
        #
        accuracy = 100 - mape
        results = {"time" : run_time, "error" : mean_error, "accuracy" : accuracy, "n_trees" : n_trees, "n_snps": n_snps}
        return results #dict

def make_predictions(x, y, model, split, analysis):
        predictions = model.predict(x)
        var_expl = sklearn.metrics.r2_score(y, predictions)
        print(str(split) + str(analysis) + " R2 is : ", + var_expl)
        return predictions, var_expl


def make_scatter(y, predictions, split, analysis):
        font = {'size': 12}
        xs = scaler.inverse_transform(y)
        ys = scaler.inverse_transform(predictions)
        xs = xs.astype(np.float32); ys = ys.astype(np.float32)
        sns.regplot(x=xs, y=ys, line_kws={"color": "orange"})
        plt.title('Scatter Plot of Predicted and Actual Values \n ($R^2$: %.2f)' % sklearn.metrics.r2_score(y, predictions))
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        xticks, xticklabels = plt.xticks() #needed to stop edge points being cut off
        # shift half a step to the left
        # x0 - (x1 - x0) / 2 = (3 * x0 - x1) / 2
        xmin = (3*xticks[0] - xticks[1])/2.
        # shift half a step to the right
        xmax = (3*xticks[-1] - xticks[-2])/2.
        plt.xlim(xmin, xmax)
        plt.xticks(xticks, rotation='vertical')
        plt.grid(True)
        plt.tight_layout()
        myfigname = str(analysis) + '_' + str(snps) + '_grid_regress_' +str(split) + '_' + str(num) + '.png'
        plt.savefig(myfigname, dpi=300, bbox_inches='tight')
        print('%s saved to file!' % myfigname)
        plt.clf()
        plt.close()


def baseline(x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model

def get_predictions(model, name):
        train_predictions, train_r2 = make_predictions(x_train, y_train, model, snps, ('train_' + str(name)))
        test_predictions, test_r2 = make_predictions(x_test, y_test, model, snps, ('test_' + str(name)))
        make_scatter(y_train, train_predictions, snps, ('train_' + str(name)))
        make_scatter(y_test, test_predictions, snps, ('test_' + str(name)))
	
test_nmae_results = ['split0_test_neg_mean_absolute_error', 'split1_test_neg_mean_absolute_error', 'split2_test_neg_mean_absolute_error','split3_test_neg_mean_absolute_error','split4_test_neg_mean_absolute_error','split5_test_neg_mean_absolute_error','split6_test_neg_mean_absolute_error','split7_test_neg_mean_absolute_error','split8_test_neg_mean_absolute_error','split9_test_neg_mean_absolute_error']
test_r2_results = ['split0_test_r2','split1_test_r2','split2_test_r2','split3_test_r2','split4_test_r2','split5_test_r2','split6_test_r2','split7_test_r2','split8_test_r2','split9_test_r2']

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

x_train, y_train = load_data(data)
name_list = np.loadtxt(data, skiprows=1, usecols=(0,), dtype='str')

scaler = preprocessing.StandardScaler().fit(y_train)
#pickle.dump(scaler, open('scaler.pkl', 'wb'))
#scaler = pickle.load(open('scaler.pkl', 'rb'))

y_train = scaler.transform(y_train)

n_snps = x_train.shape[1]
my_cv = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
#################################################SVM####SVM#####SVM####################################################################


print("Performing SVM")
c_param = [1,2]
gamma_param = [float(x) for x in np.linspace(0.1, 1, 4)]


epsilon_param = [float(x) for x in np.linspace(0.1, 1, 4)]
loss_param = ['epsilon_insensitive', 'squared_epsilon_insensitive']
kernel_param = ['poly']
degree = [1,2,3]
svm_random_grid = {'gamma':gamma_param, 'C':c_param,'kernel':kernel_param, "degree":degree}
print(svm_random_grid)
svm_random_grid2 = {'C' : c_param, 'loss':loss_param}
print(svm_random_grid2)
SVM_NCV = NestedCV(name_list = name_list, model=LinearSVR(), params_grid=svm_random_grid2, outer_kfolds=2, inner_kfolds=2, n_jobs = 2,cv_options={'randomized_search':True, 'randomized_search_iter':2, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':sklearn.metrics.r2_score, 'metric_score_indicator_lower':False})
SVM_NCV.fit(x_train, y_train.ravel(), name_list=name_list)

def ncv_results(analysis, ncv_object):
	print("Best Params of %s is %s " % (analysis, ncv_object.best_params))
	print("Outer scores of %s is %s " % (analysis, ncv_object.outer_scores))
	print("Variance of %s is %s " % (analysis, ncv_object.variance))
	with open('NCV_' + str(analysis) + '.pkl', 'wb') as ncvfile: #with open("fname.pkl", 'rb') as ncvfile:
		pickle.dump(ncv_object, ncvfile) #ncv_object = pickle.load(ncvfile)
	
ncv_results('SVM', SVM_NCV)	

