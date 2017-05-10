import os
import functools
import pdb

from collections import OrderedDict

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import preprocessing, cross_validation, metrics
from sklearn.cross_validation import StratifiedKFold, cross_val_score


def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(21, input_dim=21, init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
    
def kerasNet(features_train, features_test, labels_train, labels_test):
    seed = 7
    np.random.seed(7)
    
    #use partial on create_baseline to create all sort of estimator paramater combos
    
    #estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
    #kfold = StratifiedKFold(y=y, n_folds=10, shuffle=True, random_state=seed)
    #results = cross_val_score(estimator, X, y, cv=kfold)
    #print "Results: %d (%d)" % (results.mean()*100, results.std()*100)
    
    model = create_baseline()
    history = model.fit(features_train.values, labels_train.values, nb_epoch=50, verbose=1)
    z = model.predict(features_test.values)
    
    p1 = model.predict_proba(features_test.values)
    lloss = metrics.log_loss(labels_test.values, p1)
    #accuracy = metrics.accuracy_score(labels_test, z, normalize=True, sample_weight=None)
    
    # print log loss
    print "Model Evaluation"
    print "\tLog Loss:", lloss
    #print "\tScore:", accuracy   
    return model
    