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

 
def gradientBoosting(features_train, features_test, labels_train, labels_test):
    model = ensemble.GradientBoostingClassifier()
    model = model.fit(features_test, labels_test)

    # Z is numpy ndarray. Add, Z, to data frame as last column
    z = model.predict(features_test)
        
    # class probabilities and log loss
    p1 = model.predict_proba(features_test)
    lloss = metrics.log_loss(labels_test, p1)
    accuracy = metrics.accuracy_score(labels_test, z, normalize=True, sample_weight=None)
    
    # print log loss
    print "Model Evaluation"
    print "\tLog Loss:", lloss
    print "\tScore:", accuracy   
    return model
    
    # Get Feature Importance from the classifier
    print "Feature Importances"
    feature_importance = model.feature_importances_
    #feature_importance = feature_importance = 100.0 * (feature_importance / feature_importance.max())
    feature_importance = np.array(zip(df.columns.tolist(), feature_importance))
    feature_importance = feature_importance[np.argsort(feature_importance[:, 1])]
    for feature in feature_importance:
        print feature
        
    return model    

        
def elasticNet(features_train, features_test, labels_train, labels_test):
    l1_ratios=[.1, .5, .7, .9, .95, .99, 1]
    alpha_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    model = linear_model.ElasticNetCV(l1_ratio=l1_ratios, 
                                      #alphas=alpha_values,
                                      cv=10,
                                      n_jobs=-1,
                                      fit_intercept=True,
                                      selection = 'cyclic',)
    
    
    
    model = model.fit(features_test, labels_test)
    
    discard = 0.05
    residuals = np.abs(labels_test - model.predict(features_test))
    thresh = residuals.quantile(1 - discard)
    labels_test = labels_test.drop(features_test.index[np.where(residuals > thresh)])
    features_test = features_test.drop(features_test.index[np.where(residuals > thresh)])
    model = model.fit(features_test, labels_test)
       
    # Z is numpy ndarray. Add, Z, to data frame as last column
    z = model.predict(features_test)
    p1 = np.round(z).astype(int)
    
    # class probabilities and log loss
    lloss = metrics.log_loss(labels_test, p1)
    accuracy = metrics.accuracy_score(labels_test, p1, normalize=True, sample_weight=None)
    
    # print log loss
    print "Model Evaluation"
    print "\tLog Loss:", lloss
    print "\tScore:", accuracy
    
    return model

    
def logisticRegression(features_train, features_test, labels_train, labels_test):
    #c_values = list(np.concatenate([np.arange(x,10*x,x) for x in [0.001, 0.01, 0.1, 1, 10, 100]]))
    c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    model = linear_model.LogisticRegressionCV(Cs=c_values, 
                                             intercept_scaling=1,
                                             class_weight="balanced",
                                             cv=10,
                                             dual=False, 
                                             fit_intercept=True,
                                             penalty='l2',
                                             solver="liblinear",
                                             scoring = 'log_loss',
                                             n_jobs=-1,
                                             refit=True,
                                             tol=0.0001)
    
    model = model.fit(features_train, labels_train)
    
    # Z is numpy ndarray. Add, Z, to data frame as last column
    z = model.predict(features_test)
        
    # class probabilities and log loss
    p1 = model.predict_proba(features_test)
    lloss = metrics.log_loss(labels_test, p1)
    accuracy = metrics.accuracy_score(labels_test, z, normalize=True, sample_weight=None)
    
    # print log loss
    print "Model Evaluation"
    print "\tLog Loss:", lloss
    print "\tScore:", accuracy   
    return model
    
    
def randomForest(features_train, features_test, labels_train, labels_test):
    # grid search parameters
    parameters = {
       'n_estimators': [ 20,25 ],
       'random_state': [ 0 ],
       'max_features': [ 2 ],
       'min_samples_leaf': [150,200,250]
       }
    
    model_naive = ensemble.RandomForestClassifier(n_jobs=-1)
    model = GridSearchCV(estimator=model_naive, param_grid=parameters)
    model.fit(features_train, labels_train)

    # Z is numpy ndarray. Add, Z, to data frame as last column
    z = model.predict(features_test)
        
    # class probabilities and log loss
    p1 = model.predict_proba(features_test)
    lloss = metrics.log_loss(labels_test, p1)
    accuracy = metrics.accuracy_score(labels_test, z, normalize=True, sample_weight=None)
    
    # print log loss
    print "Model Evaluation"
    print "\tLog Loss:", lloss
    print "\tScore:", accuracy
    
    """
    # Get Feature Importance from the classifier
    feature_importance = model.feature_importances_
    feature_importance = feature_importance = 100.0 * (feature_importance / feature_importance.max())
    feature_importance = np.array(zip(df.columns.tolist(), feature_importance))
    feature_importance = feature_importance[np.argsort(feature_importance[:, 1])]
    #print "Feature Importances"
    #for feature in feature_importance:
    #    print feature
    """   
    return model

    
def supportVectors(df):   
    X = df.iloc[:,0:-1]
    y = df['target']
       
    model = svm.SVC(C=1.0,
                    kernel='linear',
                    degree=3,
                    gamma='auto',
                    probability=True,
                    cache_size=500,
                    class_weight='balanced',
                    decision_function_shape='ovr',
                    )
    model = model.fit(X, y)

    # Z is numpy ndarray. Add, Z, to data frame as last column
    df['Z'] = model.predict(X)
  
    # class probabilities and log loss
    p1 = model.predict_proba(X)
    lloss = metrics.log_loss(y, p1)
    df['p1'] = p1[:,0]
    df['p2'] = p1[:,1]

    
    # print ground truth, prediction, prob class 0, prob class 1
    #print df.iloc[:,-4:].head(10)
    
    # print log loss
    #print "Model Evaluation: Log Loss: ", lloss
    print model.score(X, y)
    
    return model 
    
     
if __name__=="__main__":
    main()