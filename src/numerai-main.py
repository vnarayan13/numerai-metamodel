import os
import functools
import pdb 

from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn import neighbors, linear_model, tree, svm, ensemble
from sklearn import preprocessing, cross_validation, metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.grid_search import GridSearchCV

from kerasneuralnetmodel import *
from xgboostmodel import *
from sklearnmodels import *


def main():
    predictions = OrderedDict()
    model_objects = OrderedDict()
    model_functions = { 
                       'XGBoost': xgb1,
                       #'Random Forest': randomForest, 
                       'Logistic Regression': logisticRegression,
                       #'Gradient Boosting': gradientBoosting,
                       #'Support Vector Machines': supportVectors,
                       'Elastic Net': elasticNet,
                       #'Neural Net': kerasNet,
                       }
    
    # Train models
    for model_name in model_functions:
        print '\nComputing', model_name
        df = pd.read_csv('numerai_datasets/numerai_training_data.csv')
        
        if model_name in ['XGBoost', 'Neural Net']:
            model_objects[model_name] =  model_functions[model_name](df)
        else:
            # use cross-validation object
            features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(df.iloc[:,0:-1], df['target'], test_size=0.3, random_state=0)   
            model_objects[model_name] = model_functions[model_name](features_train, features_test, labels_train, labels_test)
    
    # Test models
    for model_name, model in model_objects.items():
        predictions[model_name] = test(model_name, model)
        
    # Model Rank averaging and writing
    df_pred = pd.concat(predictions, keys=predictions.keys(), axis=1)
    df2 = pd.read_csv('numerai_datasets/numerai_tournament_data.csv')
    df2['probability'] = df_pred.mean(axis=1)
    
    df2[['t_id','probability']].to_csv('predictions.csv', index=False)
    
    
def test(model_name, model): 
    df2 = pd.read_csv('numerai_datasets/numerai_tournament_data.csv')
    id = df2['t_id']
    X = df2[[x for x in df2.columns if x not in ['t_id']]]    
    if model_name in ['Neural Net']: X = X.values
    
    try:
        # classifier output
        df2['Z'] = model.predict(X)           
        # class probabilities
        test_predprob = model.predict_proba(X)
        df2['p0'] = test_predprob[:,0]
        df2['p1'] = test_predprob[:,1]
    except:
        # regression output
        yhat = np.round(model.predict(X)).astype(int)
        df2['Z'] = 0.0
        df2['p0'] = 1.0-yhat
        df2['p1'] = yhat
    
    print '\n\nTest Dataset Output - %s' % model_name
    print df2[['t_id','Z','p0','p1']].head(10)

    #output = df2[['t_id','p1']]
    #output.columns = ['t_id', 'probability']  
    output = df2['p1']
    return output
    
     
if __name__=="__main__":
    main()