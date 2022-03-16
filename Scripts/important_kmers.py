 # -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 2019

@author: Alexandre Maciel Guerra
"""

import numpy as np
import pandas as pd
import sys
import pickle

from sklearn.feature_selection import chi2, SelectFromModel
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category= FutureWarning)
simplefilter(action='ignore', category= UserWarning)
simplefilter(action='ignore', category= DeprecationWarning)

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
    samples = np.array(antibiotic_df[antibiotic_df.columns[0]])
    
    n_lines = antibiotic_df.shape[0]    
    delimiter = ' '
    
    # Load kmers txt file
    update_progress(0)
    for n, line in enumerate(open(name_dataset+'_Kmer_data.txt','r')):
        if n == 0:
            dummy = np.array(line.split(delimiter), dtype=float)
            n_columns = dummy.shape[0]
            data_txt = np.zeros((n_lines, n_columns), dtype=float)
            data_txt[n,:] = dummy
        else:
            data_txt[n,:] = np.array(line.split(delimiter), dtype=float)
        update_progress((n+1)/n_lines)
    

    print(antibiotic_df.columns[1:])
    for name_antibiotic in antibiotic_df.columns[1:]:
        print("Antibiotic: {}".format(name_antibiotic))

        target_str = np.array(antibiotic_df[name_antibiotic])
        
        target = np.zeros(len(target_str)).astype(int)
        idx_S = np.where(target_str == 'S')[0]
        idx_R = np.where(target_str == 'R')[0]
        idx_I = np.where((target_str != 'R') & (target_str != 'S'))[0]
        target[idx_R] = 1
        
        idx = np.hstack((idx_S,idx_R))
        
        if len(idx) == 0:
            print("Empty")
            continue
        
        target = target[idx]

        # Skip antibiotic if the number of samples is too small
        count_class = Counter(target)
        print(count_class)
        if count_class[0] < 12 or count_class[1] < 12:
            continue 

        if method == "model":
            scores, pvalue = chi2(data_txt[idx,:], target)
            scores[np.isnan(pvalue)] = 0
            pvalue[np.isnan(pvalue)] = 1
            cols = np.where(pvalue < 0.05)[0]

            ixgrid = np.ix_(idx,cols)
            model = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50)).fit(data_txt[ixgrid], target)
            data = model.transform(data_txt[ixgrid])
            coef = model.estimator_.feature_importances_
            cols_model = model.get_support(indices=True)

            print("Optimal number of features : %d" % data.shape[1])
            print("Threshold = {}".format(model.threshold_))

            results_array = np.zeros((len(cols_model),4))
            results_array[:,0] = cols[cols_model]
            results_array[:,1] = scores[cols[cols_model]]
            results_array[:,2] = pvalue[cols[cols_model]]
            results_array[:,3] = coef[cols_model]
        
            with open("data_"+method+"_"+name_dataset+"_"+name_antibiotic+'.pickle', 'wb') as f:
                pickle.dump(data, f)
        
        np.savetxt(results_folder+"/features_"+method+"_"+name_dataset+'_'+name_antibiotic+'.txt', cols[cols_model], fmt='%d')    
        np.savetxt(results_folder+"/"+name_dataset+"_"+name_antibiotic+"_"+method+"_pvalue.csv", results_array, delimiter=",")