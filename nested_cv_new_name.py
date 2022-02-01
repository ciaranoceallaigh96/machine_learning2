import logging as log
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.utils.multiclass import type_of_target
from joblib import Parallel, delayed
import os.path
import time
import subprocess
import sys
import sklearn
from sklearn import preprocessing
import os
import copy
print(os.getcwd())
print("WARNING THIS IS AN EDITED SCRIPT - Ciaran Kelly 2021 \n Edited with permission under liscence \n Flex version")
#set_size = 10006    
#print("Set size set to %s" % set_size)

def load_data(data, set_size, organism='arabidopsis'):
        print("Set size set to %s" % set_size)
        dataset = np.loadtxt(data, skiprows=1, dtype='str')
        if organism=='mouse':
            print("organism: %s" % organism)
            x = dataset[: , 6:(set_size+6)].astype(np.int)
            #x = np.where(x==0, 3, x) ; x = np.where(x==2, 0, x) ; x = np.where(x==3, 2, x) #flipping around the alleles
        else:
            print("organism: %s" % organism)
            x = dataset[: , 6:(set_size+6)].astype(np.int)/2
        y = dataset[: , 5 ].astype(np.float)
        y = y.reshape(-1,1)
        #print("Performing split of raw data....")
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
        return x, y #x_train, y_train, x_test, y_test


def bash_script(train_index, test_index, train_names, test_names, outer_count, inner_count, phenfile, set_size, snps, organism='arabidopsis', outer=False):
        print(os.getcwd())
        if outer==True:
            foo='out'
        else:
            foo='in'
        if not os.path.exists('train_raw_plink_' + str(snps) + '_' + str(outer_count) + '_in_' + str(inner_count) + '_' + foo + '.raw'):
            print("SETTING OFF CUSTOM BASH SCRIPT")
            with open("name_vector_train.txt", 'w') as f:
                for item in train_names:
                    f.write("%s %s\n" % (item, item))
            with open("name_vector_test.txt", 'w') as f:
                for item in test_names:
                    f.write("%s %s\n" % (item, item))        

            if organism != 'mouse':
                subprocess.run(["/external_storage/ciaran/machine_learning2/bash_script.sh", str(outer_count), str(inner_count), foo, str(phenfile), str(set_size), str(snps)]) 
            else:
                subprocess.run(["/external_storage/ciaran/machine_learning2/bash_script_mouse.sh", str(outer_count), str(inner_count), foo, str(phenfile), str(set_size), str(snps)])
        #while not os.path.exists('train_raw_plink_shuf_' + str(outer_count) + '_in_' + str(inner_count) + '.raw'):
        while not os.path.exists('test_raw_plink_' +  str(snps) +  '_' + str(outer_count) + '_in_' + str(inner_count) + '_' + foo + '.raw'):
            time.sleep(20)
        print('test_raw_plink_' +  str(snps) + '_' + str(outer_count) + '_in_' + str(inner_count) + '_' + foo + '.raw')
        new_X_train , new_y_train = load_data('train_raw_plink_' +  str(snps) + '_' + str(outer_count) + '_in_' + str(inner_count) + '_' + foo + '.raw', set_size, organism) #made from bash_script.sh
        new_X_test , new_y_test  = load_data('test_raw_plink_' +  str(snps) + '_' + str(outer_count) + '_in_' + str(inner_count) + '_' + foo + '.raw', set_size, organism)
        scaler = preprocessing.StandardScaler().fit(new_y_train)
        new_y_train = scaler.transform(new_y_train)
        new_y_test = scaler.transform(new_y_test)
        return new_X_train, new_X_test, new_y_train, new_y_test


class NestedCV():
    '''A general class to handle nested cross-validation for any estimator that
    implements the scikit-learn estimator interface.

    Parameters
    ----------
    model : estimator
        The estimator implements scikit-learn estimator interface.

    params_grid : dict
        The dict contains hyperparameters for model.

    outer_kfolds : int
        Number of outer K-partitions in KFold

    inner_kfolds : int
        Number of inner K-partitions in KFold
    n_jobs : int
        Number of jobs to run in parallel

    cv_options: dict, default = {}
        Nested Cross-Validation Options, check docs for details.

        metric : callable from sklearn.metrics, default = mean_squared_error
            A scoring metric used to score each model

        metric_score_indicator_lower : boolean, default = True
            Choose whether lower score is better for the metric calculation or higher score is better,
            `True` means lower score is better.

        sqrt_of_score : boolean, default = False
            Whether or not the square root should be taken of score

        randomized_search : boolean, default = False
            Whether to use gridsearch or randomizedsearch from sklearn

        randomized_search_iter : int, default = 10
            Number of iterations for randomized search

        recursive_feature_elimination : boolean, default = False
            Whether to do recursive feature selection (rfe) for each set of different hyperparameters
            in the inner most loop of the fit function.

        rfe_n_features : int, default = 1
            If recursive_feature_elimination is enabled, select n number of features
        
        predict_proba : boolean, default = False
            If true, predict probabilities instead for a class, instead of predicting a class
        
        multiclass_average : string, default = 'binary'
            For some classification metrics with a multiclass prediction, you need to specify an
            average other than 'binary'
    '''

    def __init__(self, model_name, name_list, model, params_grid, goal_dict, time_dict, outer_kfolds, inner_kfolds, n_jobs = 1, cv_options={}):
        self.model_name = model_name
        self.name_list = name_list
        self.model = model
        self.params_grid = params_grid
        self.outer_kfolds = outer_kfolds
        self.inner_kfolds = inner_kfolds
        self.n_jobs = n_jobs
        self.metric = cv_options.get('metric', mean_squared_error)
        self.metric_score_indicator_lower = cv_options.get(
            'metric_score_indicator_lower', True)
        self.sqrt_of_score = cv_options.get('sqrt_of_score', False)
        self.randomized_search = cv_options.get('randomized_search', False)
        self.randomized_search_iter = cv_options.get(
            'randomized_search_iter', 10)
        self.recursive_feature_elimination = cv_options.get(
            'recursive_feature_elimination', False)
        self.rfe_n_features = cv_options.get(
            'rfe_n_features', 0)
        self.predict_proba = cv_options.get(
            'predict_proba', False)
        self.multiclass_average = cv_options.get(
            'multiclass_average', 'binary')
        self.outer_scores = []
        self.best_params = {}
        self.best_inner_score_list = []
        self.variance = []

    # to check if use sqrt_of_score and handle the different cases
    def _transform_score_format(self, scoreValue):
        if self.sqrt_of_score:
            return np.sqrt(scoreValue)
        return scoreValue

    # to convert array of dict to dict with array values, so it can be used as params for parameter tuning
    def _score_to_best_params(self, best_inner_params_list):
        params_dict = {}
        for best_inner_params in best_inner_params_list:
            for key, value in best_inner_params.items():
                if key in params_dict:
                    if value not in params_dict[key]:
                        params_dict[key].append(value)
                else:
                    params_dict[key] = [value]
        return params_dict

    # a function to handle recursive feature elimination
    def _fit_recursive_feature_elimination(self, X_train_outer, y_train_outer, X_test_outer):
        rfe = RFECV(estimator=self.model,
                    min_features_to_select=self.rfe_n_features, cv=self.inner_kfolds, n_jobs = self.n_jobs)
        rfe.fit(X_train_outer, y_train_outer)
        
        log.info('Best number of features was: {0}'.format(rfe.n_features_))

        # Assign selected features to data
        return rfe.transform(X_train_outer), rfe.transform(X_test_outer)
    
    def _predict_and_score(self, X_test, y_test):
        #XXX: Implement type_of_target(y)
        
        if(self.predict_proba):
            y_type = type_of_target(y_test)
            if(y_type in ('binary')):
                pred = self.model.predict_proba(X_test)[:,1]
            else:
                pred = self.model.predict_proba(X_test)
                
        else:
            pred = self.model.predict(X_test)
        
        if(self.multiclass_average == 'binary'):
            print(self.metric(y_test, np.nan_to_num(pred))) #added in nan_to_num because3 of error
            return self.metric(y_test, np.nan_to_num(pred)), np.nan_to_num(pred)
        else:
            print(self.metric(y_test, pred, average=self.multiclass_average))
            return self.metric(y_test, pred, average=self.multiclass_average), pred
    def _best_of_results(self, results):
        best_score = None
        best_parameters = {}
        
        for score_parameter in results:
            if(self.metric_score_indicator_lower):
                if(best_score == None or score_parameter[0] < best_score):
                    best_score = score_parameter[0]
                    best_parameters = score_parameter[1]
            else:
                if(best_score == None or score_parameter[0] > best_score):
                    best_score = score_parameter[0]
                    best_parameters = score_parameter[1]
        
        return best_score, best_parameters

    def _parallel_fitting(self, X_train_inner, X_test_inner, y_train_inner, y_test_inner, param_dict):
                    log.debug(
                        '\n\tFitting these parameters:\n\t{0}'.format(param_dict))
                    # Set hyperparameters, train model on inner split, predict results.
                    self.model.set_params(**param_dict)
                    print("Blue")
                    # Fit model with current hyperparameters and score it
                    if(type(self.model).__name__ == 'KerasRegressor' or type(self.model).__name__ == 'KerasClassifier' or type(self.model).__name__ == 'Pipeline'):
                        self.model.fit(X_train_inner, y_train_inner, validation_data=(X_test_inner, y_test_inner)) #will allow for learning curve plotting
                    else:
                        self.model.fit(X_train_inner, y_train_inner)
                    print("Red")
                    # Predict and score model
                    inner_grid_score, inner_pred = self._predict_and_score(X_test_inner, y_test_inner.ravel())
                    #inner_grid_train_score = self.model.score(X_train_inner, y_train_inner) #to check stability 
                    inner_train_pred = self.model.predict(X_train_inner)
                    inner_grid_train_score = self.metric(y_train_inner, np.nan_to_num(inner_train_pred)) #had to replace one line inner_grid_train_score because of NA,Infiinity error, this allows NAs in train_pred to be dealt with
                    # Cleanup for Keras
                    print("Typer", str(type(self.model).__name__))
                    if(type(self.model).__name__ == 'KerasRegressor' or
                       type(self.model).__name__ == 'KerasClassifier' or type(self.model).__name__ == 'Pipeline'):
                        print("Hello world")
                        from tensorflow.keras import backend as K
                        K.clear_session()

                    return self._transform_score_format(inner_grid_score), param_dict, inner_grid_train_score

    def fit(self, X, y, name_list, model_name, goal_dict, time_dict, phenfile, set_size, snps, organism):
        '''A method to fit nested cross-validation 
        Parameters
        ----------
        X : pandas dataframe (rows, columns)
            Training dataframe, where rows is total number of observations and columns
            is total number of features

        y : pandas dataframe
            Output dataframe, also called output variable. y is what you want to predict.

        Returns
        -------
        It will not return directly the values, but it's accessable from the class object it self.
        You should be able to access:

        variance
            Model variance by numpy.var()

        outer_scores 
            Outer score List.

        best_inner_score_list 
            Best inner scores for each outer loop

        best_params 
            All best params from each inner loop cumulated in a dict

        best_inner_params_list 
            Best inner params for each outer loop as an array of dictionaries
        '''
        
        log.debug(
            '\n{0} <-- Running this model now'.format(type(self.model).__name__))

        self.X = X
        self.y = y
        self.model_name = model_name
        self.name_list = name_list 
        self.phenfile = phenfile
        self.set_size = set_size
        self.goal_dict = goal_dict
        self.time_dict = time_dict
        self.snps = snps
        # If Pandas dataframe or series, convert to array
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
        if(self.randomized_search):
            param_func = ParameterSampler(param_distributions=self.params_grid,
                                                   n_iter=self.randomized_search_iter)
        else:
            param_func = ParameterGrid(param_grid=self.params_grid)
        
        outer_cv = KFold(n_splits=self.outer_kfolds, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=self.inner_kfolds, shuffle=True, random_state=42)
        stability_dict = copy.deepcopy(goal_dict) #fully independent copy of an object 
        outer_count = 1
        outer_scores = []
        variance = []
        best_inner_params_list = []  # Change both to by one thing out of key-value pair
        best_inner_score_list = []
        print("ADBUCE")
        # Split X and y into K-partitions to Outer CV
        for (i, (train_index, test_index)) in enumerate(outer_cv.split(X, y)):
            log.debug(
                '\n{0}/{1} <-- Current outer fold'.format(i+1, self.outer_kfolds))
            X_train_outer, X_test_outer = X[train_index], X[test_index]
            y_train_outer, y_test_outer = y[train_index], y[test_index]
            outer_train_names, outer_test_names = name_list[train_index], name_list[test_index]
            best_inner_params = {}
            best_inner_score = None
            search_scores = []
            inner_count = 0
            outer=False
            # Split X_train_outer and y_train_outer into K-partitions to be inner CV
            for (j, (train_index_inner, test_index_inner)) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
                log.debug(
                    '\n\t{0}/{1} <-- Current inner fold'.format(j+1, self.inner_kfolds))
                #X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
                #y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]
                inner_count += 1 ; print("INNER COUNT NO. ", str(inner_count))
                inner_train_names, inner_test_names = outer_train_names[train_index_inner], outer_train_names[test_index_inner]
                X_train_inner, X_test_inner, y_train_inner, y_test_inner = bash_script(train_index_inner, test_index_inner, inner_train_names, inner_test_names, outer_count, inner_count, phenfile, set_size, snps, organism, outer=False)
                if model_name == 'CNN':
                    X_train_inner = X_train_inner.reshape(X_train_inner.shape[0],X_train_inner.shape[1],1)
                    X_test_inner = X_test_inner.reshape(X_test_inner.shape[0],X_test_inner.shape[1],1)
                if self.recursive_feature_elimination:
                        X_train_inner, X_test_inner = self._fit_recursive_feature_elimination(
                                    X_train_inner, y_train_inner, X_test_inner)
                

                results = []
                
                for parameters in param_func:
                  print(parameters)
                  tic = time.clock()
                  inner_grid_score, param_dictionary,inner_train_score = self._parallel_fitting(X_train_inner, X_test_inner,y_train_inner.ravel()-1, y_test_inner.ravel()-1,param_dict=parameters)
                  toc = time.clock()
                  results.append((inner_grid_score,param_dictionary))
                  for key in param_dictionary:
                    time_dict[key][param_dictionary[key]].append(toc-tic)
                    goal_dict[key][param_dictionary[key]].append(inner_grid_score)
                    if inner_train_score > 0.1:
                      if inner_grid_score <= 0:
                        inner_grid_score = 0
                      stability_dict[key][param_dictionary[key]].append(inner_train_score-inner_grid_score)
                    else:
                      stability_dict[key][param_dictionary[key]].append(1)
                search_scores.extend(results)
            
            best_inner_score, best_inner_params = self._best_of_results(search_scores)
            
            best_inner_params_list.append(best_inner_params)
            best_inner_score_list.append(best_inner_score)
            #inner_count = 0
            print("OUTER COUNT NO. ", str(outer_count))
            # Fit the best hyperparameters from one of the K inner loops
            self.model.set_params(**best_inner_params)
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = bash_script(train_index, test_index, outer_train_names, outer_test_names, outer_count, inner_count, phenfile, set_size, snps, organism, outer=True)
            outer_count += 1
            if model_name == 'CNN':
                X_train_outer = X_train_outer.reshape(X_train_outer.shape[0],X_train_outer.shape[1],1)
                X_test_outer = X_test_outer.reshape(X_test_outer.shape[0],X_test_outer.shape[1],1)
            if(type(self.model).__name__ == 'KerasRegressor' or type(self.model).__name__ == 'KerasClassifier' or type(self.model).__name__ == 'Pipeline'):
                result = self.model.fit(X_train_outer, y_train_outer.ravel(), validation_data=(X_test_inner, y_test_inner))
                object = result.model.history
                plt.plot(object.history['loss'])
                plt.plot(object.history['val_loss'])
                plt.title('Model Loss Curve'); plt.ylabel('Mean Absolute Error'); plt.xlabel('Epoch'); plt.legend(['Train', 'Test'], loc='upper left')
                plt.show()
                plt.savefig('loss_training_curve_' + str(outer_count-1) + '_' + model_name, dpi=300)
                plt.clf() ; plt.close() #coeff_determination
                plt.plot(object.history['coeff_determination'])
                plt.plot(object.history['val_coeff_determination'])
                plt.title('Model $R^{2}$ Curve'); plt.ylabel('$R^{2}$'); plt.xlabel('Epoch'); plt.legend(['Train', 'Test'], loc='upper left')
                plt.show()
                plt.savefig('loss_r2_curve_' + str(outer_count-1) + '_' + model_name, dpi=300)
                plt.clf() ; plt.close() #coeff_determination
                
            else:
                self.model.fit(X_train_outer, y_train_outer.ravel())
            # Get score and prediction
            score,pred = self._predict_and_score(X_test_outer, y_test_outer.ravel())
            outer_scores.append(self._transform_score_format(score))

            # Append variance
            variance.append(np.var(pred, ddof=1))

            log.debug('\nResults for outer fold:\nBest inner parameters was: {0}'.format(
                best_inner_params_list[i]))
            log.debug('Outer score: {0}'.format(outer_scores[i]))
            log.debug('Inner score: {0}'.format(best_inner_score_list[i]))
         
        self.variance = variance
        self.outer_scores = outer_scores
        self.best_inner_score_list = best_inner_score_list
        self.best_params = self._score_to_best_params(best_inner_params_list)
        self.best_inner_params_list = best_inner_params_list
        self.goal_dict = goal_dict
        self.time_dict = time_dict
        self.stability_dict = stability_dict

    # Method to show score vs variance chart. You can run it only after fitting the model.
    def score_vs_variance_plot(self):
        # Plot score vs variance
        plt.figure()
        plt.subplot(211)

        variance_plot, = plt.plot(self.variance, color='b')
        score_plot, = plt.plot(self.outer_scores, color='r')

        plt.legend([variance_plot, score_plot],
                   ["Variance", "Score"],
                   bbox_to_anchor=(0, .4, .5, 0))

        plt.title("{0}: Score VS Variance".format(type(self.model).__name__),
                  x=.5, y=1.1, fontsize="15")
