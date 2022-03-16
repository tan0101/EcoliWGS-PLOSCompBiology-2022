# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:40:56 2019

@author: psxam11
"""

import numpy as np
import pandas as pd
import sys
import pickle


from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_validate
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter
from pathlib import Path

# import warnings filter
from warnings import simplefilter
import warnings
# ignore all future warnings
simplefilter(action='ignore', category= FutureWarning)
simplefilter(action='ignore', category= UserWarning)
simplefilter(action='ignore', category= DeprecationWarning)

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn),
           'auc': 'roc_auc',
           'acc': make_scorer(accuracy_score),
           'kappa': make_scorer(cohen_kappa_score)}

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()  

def classification(data, target):
    # Nested Cross Validation:
    inner_loop_cv = 3
    outer_loop_cv = 5
    
    # Number of random trials:
    NUM_TRIALS = 3
    
    # Initialize Variables:
    scores_auc_test = np.zeros(NUM_TRIALS)
    scores_auc_train = np.zeros(NUM_TRIALS)
    
    # Grid of Parameters:
    SVC_grid = {"clf__gamma": [0.0001, 0.001, 0.01, 0.1], "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        
    # Classifiers:
    names = ["RBF SVM"]
    classifiers = [SVC()]
    
    # Standardize data: zero mean and unit variance
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    count_class = Counter(target)
    
    if count_class[0] < 12 or count_class[1] < 12:
        return np.mean(scores_auc_test), np.mean(scores_auc_train)
    
    # Loop for each trial
    for i in range(NUM_TRIALS):
    
        inner_cv = StratifiedKFold(n_splits=inner_loop_cv, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=outer_loop_cv, shuffle=True, random_state=i)
    
        for name, clf in zip(names, classifiers):
            model = Pipeline([
                ('clf', clf)
            ])
            
            # Inner Search
            classif = GridSearchCV(estimator=model, param_grid=SVC_grid, cv=inner_cv)
            classif.fit(data, target)
            
            # Outer Search
            cv_results = cross_validate(classif, data, target, scoring="roc_auc", cv=outer_cv, return_train_score=True)
                
            scores_auc_test[i] = cv_results['test_score'].mean()
            scores_auc_train[i] = cv_results['train_score'].mean()
            
    return np.mean(scores_auc_test), np.mean(scores_auc_train)

if __name__ == "__main__":
    method = "model"
    name_dataset = "" 
    
    # Load Antibiotic Data
    antibiotic_df = pd.read_csv(name_dataset+'_AMR_data_RSI.csv', header = [0])
    
    n_lines = antibiotic_df.shape[0]
    samples = np.array(antibiotic_df[antibiotic_df.columns[0]])
        
    print(antibiotic_df.columns[1:])
    for name_antibiotic in antibiotic_df.columns[1:]:
        print("Antibiotic: {}".format(name_antibiotic))
        
        target_str = np.array(antibiotic_df[name_antibiotic])
        
        target = np.zeros(len(target_str)).astype(int)
        idx_S = np.where(target_str == 'S')[0]
        idx_R = np.where(target_str == 'R')[0]
        idx_I = np.where((target_str != 'R') & (target_str != 'S'))[0]
        target[idx_R] = 1
        target[idx_I] = 2

        idx = np.hstack((idx_S,idx_R))
        
        if len(idx) == 0:
            print("Empty")
            continue
        
        samples_name = np.delete(samples,idx_I, axis=0)
        target = target[idx]
        
        count_class = Counter(target)
        print(count_class)

        if count_class[0] < 12 or count_class[1] < 12:
            continue


        file_name = "data_model_"+name_dataset+"_"+name_antibiotic+'.pickle'
        
        my_file = Path(file_name)

        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            continue
        else:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                
        sm = SMOTE(random_state=42)
        data, target = sm.fit_resample(data, target)
        print('Resampled dataset shape %s' % Counter(target))

        n_samples = data.shape[0]    
        samples_array = np.arange(n_samples)
        results_array_test = []
        results_array_train = []
        id_samples_array = []
        while n_samples > 24:
            print(n_samples)
            scores_auc_test = np.zeros(n_samples)
            scores_auc_train = np.zeros(n_samples)
            
            update_progress(0)
            for i in range(n_samples):
                samples_array_sel = np.delete(samples_array,i)
                data_sel = data[samples_array_sel,:]
                target_sel = target[samples_array_sel]
                
                scores_auc_test[i], scores_auc_train[i] = classification(data_sel, target_sel)
                update_progress((i+1)/n_samples)
                
            id_min = np.argmin(scores_auc_test)
            id_samples_array.append(samples_array[id_min])
            samples_array = np.delete(samples_array, id_min)
            n_samples = len(samples_array)
            results_array_test.append(scores_auc_test[id_min])
            results_array_train.append(scores_auc_train[id_min])
            print("\n")
            print("Test score = {}".format(scores_auc_test[id_min]))
            print("Train score = {}".format(scores_auc_train[id_min]))
            
        print(results_array_test)
        print(results_array_train)
        np.savetxt("Synthetic_backward_elimination_test_AUC_"+name_antibiotic+".csv", results_array_test, delimiter=",")
        np.savetxt("Synthetic_backward_elimination_train_AUC_"+name_antibiotic+".csv", results_array_train, delimiter=",")
        np.savetxt("Synthetic_backward_elimination_samples_id_"+name_antibiotic+".csv", id_samples_array, delimiter=",")
            