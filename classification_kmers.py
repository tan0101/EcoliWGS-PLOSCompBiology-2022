# -*- coding: utf-8 -*-
"""
Created on Tue Oct 8 2019

@author: Alexandre Maciel Guerra
"""

import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter

# import warnings filter
from warnings import simplefilter
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

if __name__ == "__main__":
    method = "model"
    
    #Add the name of the data set
    name_dataset = ""

    #Add the folder where the results must be saved
    results_folder = ""
    
    # Load AMR profile:
    antibiotic_df = pd.read_csv(name_dataset+'_AMR_data_RSI.csv', header = [0])
    delimiter = ' '
    
    # Nested Cross Validation:
    inner_loop_cv = 3
    outer_loop_cv = 5
    
    # Number of random trials:
    NUM_TRIALS = 30
    
    # Grid of Parameters:
    C_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    est_grid = {"clf__n_estimators": [2, 4, 8, 16, 32, 64]}
    MLP_grid = {"clf__alpha": [0.001, 0.01, 0.1, 1, 10, 100], "clf__learning_rate_init": [0.001, 0.01, 0.1, 1],
        "clf__hidden_layer_sizes": [10, 20, 40, 100, 200, 300, 400, 500]}
    SVC_grid = {"clf__gamma": [0.0001, 0.001, 0.01, 0.1], "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    DT_grid = {"clf__max_depth": [10, 20, 30, 50, 100]}
    XGBoost_grid = {"clf__n_estimators": [2, 4, 8, 16, 32, 64], "clf__learning_rate": [0.001, 0.01, 0.1, 1]}
        
    # Classifiers:
    names = ["Logistic Regression", "Linear SVM", "RBF SVM",
        "Extra Trees", "Random Forest", "AdaBoost", "XGBoost",
        "Naive Bayes", "QDA", "LDA"] 

    classifiers = [
        LogisticRegression(),
        LinearSVC(loss='hinge'),
        SVC(),
        ExtraTreesClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LinearDiscriminantAnalysis()
        ]
        
    # Initialize Variables:
    scores_auc = np.zeros([NUM_TRIALS,len(classifiers)])
    scores_acc = np.zeros([NUM_TRIALS,len(classifiers)])
    scores_sens = np.zeros([NUM_TRIALS,len(classifiers)])
    scores_spec = np.zeros([NUM_TRIALS,len(classifiers)])
    scores_prec = np.zeros([NUM_TRIALS,len(classifiers)])
    scores_kappa = np.zeros([NUM_TRIALS,len(classifiers)])

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
        
        # Skip antibiotic if the number of samples is too small
        count_class = Counter(target)
        print(count_class)

        if count_class[0] < 12 or count_class[1] < 12:
            continue

        features_df = pd.read_csv(name_dataset+"_"+name_antibiotic+"_"+method+"_pvalue.csv", header = None)
        n_features = features_df.shape[0]
        
        print("Number of features = {}".format(n_features))

        features_data = np.array(features_df.loc[:,features_df.columns[0]]).astype(int)

        # Load kmers based on the chi-square test results
        data = np.zeros((n_lines,n_features))
        update_progress(0)
        for n, line in enumerate(open(name_dataset+'_Kmer_data.txt','r')):
            dummy = np.array(line.split(delimiter), dtype=float)
            data[n,:] = dummy[features_data]
            update_progress((n+1)/n_lines)

        data = data[idx,:]

        # Standardize data: zero mean and unit variance
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # Loop for each trial
        update_progress(0)
        for i in range(NUM_TRIALS):
            #print("Trial = {}".format(i))
        
            inner_cv = StratifiedKFold(n_splits=inner_loop_cv, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_loop_cv, shuffle=True, random_state=i)
        
            k = 0
        
            for name, clf in zip(names, classifiers):
                model = Pipeline([
                    ('sampling',SMOTE(random_state=i)),
                    ('clf', clf)
                ])
                
                if name == "QDA" or name == "LDA" or name == "Naive Bayes":
                    classif = model
                else:
                    if name == "RBF SVM":
                        grid = SVC_grid              
                    elif name == "Random Forest" or name == "AdaBoost" or name == "Extra Trees":
                        grid = est_grid
                    elif name == "Neural Net":
                        grid = MLP_grid
                    elif name == "Linear SVM":
                        grid = C_grid
                    elif name == "Decision Tree":
                        grid = DT_grid
                    elif name == "XGBoost":
                        grid = XGBoost_grid
                    else:
                        grid = C_grid
        
                    # Inner Search
                    classif = GridSearchCV(estimator=model, param_grid=grid, cv=inner_cv)
                    classif.fit(data, target)
                
                # Outer Search
                cv_results = cross_validate(classif, data, target, scoring=scoring, cv=outer_cv)
                    
                tp = cv_results['test_tp']
                tn = cv_results['test_tn']
                fp = cv_results['test_fp']
                fn = cv_results['test_fn']
                
                sens = np.zeros(outer_loop_cv)
                spec = np.zeros(outer_loop_cv)
                prec = np.zeros(outer_loop_cv)
                
                for j in range(outer_loop_cv):
                    TP = tp[j]
                    TN = tn[j]
                    FP = fp[j]
                    FN = fn[j]
                    
                    # Sensitivity, hit rate, recall, or true positive rate
                    sens[j] = TP/(TP+FN)
                    
                    # Fall out or false positive rate
                    FPR = FP/(FP+TN)
                    spec[j] = 1 - FPR
                    
                    #Precision
                    if TP + FP > 0:
                        prec[j] = TP / (TP + FP)
    
                scores_sens[i,k] = sens.mean()
                scores_spec[i,k] = spec.mean()
                scores_prec[i,k] = prec.mean()
                scores_auc[i,k] = cv_results['test_auc'].mean()
                scores_acc[i,k] = cv_results['test_acc'].mean()
                scores_kappa[i,k] = cv_results['test_kappa'].mean()
                
                k = k + 1
                
            update_progress((i+1)/NUM_TRIALS)

        # Save results
        results = np.zeros((12,len(classifiers)))
        scores = [scores_auc, scores_acc, scores_sens, scores_spec, scores_prec, scores_kappa]
        for counter_scr, scr in enumerate(scores):
            results[2*counter_scr,:] = np.mean(scr,axis=0)
            results[2*counter_scr + 1,:] = np.std(scr,axis=0)
            
        names_scr = ["AUC_Mean", "AUC_Std", "Acc_Mean", "Acc_Std", 
            "Sens_Mean", "Sens_Std", "Spec_Mean", "Spec_Std", 
            "Prec_Mean", "Prec_Std","Kappa_Mean", "Kappa_Std"]

        results_df=pd.DataFrame(results, columns=names, index=names_scr)
        
        np.savetxt(results_folder+"/SMOTE_"+method+"_"+name_dataset+"_"+name_antibiotic+"_auc.csv", scores_auc, delimiter=",")
        np.savetxt(results_folder+"/SMOTE_"+method+"_"+name_dataset+"_"+name_antibiotic+"_acc.csv", scores_acc, delimiter=",")
        np.savetxt(results_folder+"/SMOTE_"+method+"_"+name_dataset+"_"+name_antibiotic+"_sens.csv", scores_sens, delimiter=",")
        np.savetxt(results_folder+"/SMOTE_"+method+"_"+name_dataset+"_"+name_antibiotic+"_spec.csv", scores_spec, delimiter=",")
        np.savetxt(results_folder+"/SMOTE_"+method+"_"+name_dataset+"_"+name_antibiotic+"_prec.csv", scores_prec, delimiter=",")
        np.savetxt(results_folder+"/SMOTE_"+method+"_"+name_dataset+"_"+name_antibiotic+"_kappa.csv", scores_kappa, delimiter=",")
        results_df.to_csv(results_folder+"/SMOTE_"+method+"_"+"results_"+name_dataset+"_"+name_antibiotic+".csv")
        