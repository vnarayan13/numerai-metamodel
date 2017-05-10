import os
import functools
import pdb

from collections import OrderedDict

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn import preprocessing, cross_validation, metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.grid_search import GridSearchCV


def xgb1(df):
    #Choose all predictors except target
    predictors = df[[x for x in df.columns if x not in ['target']]]
    target = df['target']
    features = None
    
    xgb_model = XGBClassifier(learning_rate=0.1,
                         n_estimators=1000,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'binary:logistic',
                         scale_pos_weight=1,
                         seed=27)
    
    # optimize Classifier for n_estimators
    xgb_model = modelfit(xgb_model, df, predictors, target)

    """
    # with feature weighting
    #xgb_model9, features = modelfit(xgb_model, df, predictors, target, returnFeatures=20)
    #predictors = predictors[features]
    #xgb_model = modelfit(xgb_model, df, predictors, target)
    """
    
    """
    # Grid Search on parameters
    param_test1 = {'max_depth':range(3,10),
                   'min_child_weight':range(1,6)
                   }   
    param1 = xgbGridSearch(predictors, target, xgb_model, param_test1)                        
    xgb_model.set_params(max_depth=param1['max_depth'], min_child_weight=param1['min_child_weight'])
    
    # Final Model
    xgb_model = modelfit(xgb_model, df, predictors, target)
    """
        
    return xgb_model
    
    
def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, returnFeatures=None):   
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(predictors.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='logloss', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(predictors, target, eval_metric='logloss')
        
    #Predict training set:
    dtrain['Z'] = alg.predict(predictors)
    dtrain_predprob = alg.predict_proba(predictors)
    
    # logloss
    lloss = metrics.log_loss(target, dtrain_predprob)
    dtrain['p0'] = dtrain_predprob[:,0]
    dtrain['p1'] = dtrain_predprob[:,1]
    
    # print ground truth, prediction, prob class 0, prob class 1
    #print dtrain.iloc[:,-5:].head(10)
    
    print "Model Probabilities"
    # Print model report:
    print "\nModel Report"
    print "Log Loss: ", lloss
    print "Accuracy : %.4g" % metrics.accuracy_score(target.values, dtrain['Z'])
    print "AUC Score (Train): %f" % metrics.roc_auc_score(target, dtrain_predprob[:,1])
                    
    #final_params = alg.get_xgb_params()
    #print final_params
    
    if returnFeatures is not None:
        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        features = feat_imp[:returnFeatures].index.tolist()
        return alg, features
    else:    
        return alg

        
def xgbGridSearch(predictors, target, model_in, param_test):
    print 'Performing Grid Search'
    print 'Parameters:', param_test
    
    gsearch1 = GridSearchCV(estimator=model_in, 
                            param_grid = param_test, 
                            scoring='log_loss',
                            n_jobs=4,
                            iid=False,
                            cv=5)
    gsearch1.fit(predictors, target)
    
    print gsearch1.grid_scores_
    print gsearch1.best_params_
    print gsearch1.best_score_
    optimized_params = gsearch1.best_params_
    
    return optimized_params
    