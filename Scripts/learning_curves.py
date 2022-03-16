# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:40:56 2019

@author: psxam11
"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pickle

from collections import Counter
from pathlib import Path

# import warnings filter
from warnings import simplefilter
import warnings
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
    name_dataset = "" 
    fname = "SMOTE" #"Synthetic"
    
    # Locad Antibiotic Data
    antibiotic_df = pd.read_csv(name_dataset+'_AMR_data_RSI.csv', header = [0])
    
    n_lines = antibiotic_df.shape[0]
    samples = np.array(antibiotic_df[antibiotic_df.columns[0]])
        
    k=0
    fig, axes = plt.subplots(7, 3, figsize=(10, 15))
        
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
                
        n_samples = data.shape[0]
        
        file_name = fname+"backward_elimination_test_AUC_"+name_antibiotic+'.csv'
        
        my_file = Path(file_name)

        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            continue
        else:
            test_data = np.array(pd.read_csv(file_name, header=None))
            train_data = np.array(pd.read_csv(fname+"backward_elimination_train_AUC_"+name_antibiotic+'.csv', header=None))
        
        test_data = test_data[test_data>0]
        train_data = train_data[train_data>0]
        
        n_samples = len(target)-1
        
        
        if fname == "Synthetic":
            if count_class[0] > count_class[1]:
                x = np.arange(2*count_class[0]-1,2*count_class[0]-len(test_data)-1,-1)
            else:
                x = np.arange(2*count_class[1]-1,2*count_class[1]-len(test_data)-1,-1)
        else:
            x = np.arange(n_samples,n_samples-len(test_data),-1)
        
        axes = axes.ravel()
        
        axes[k].set_title(name_antibiotic)
        axes[k].set_xlabel("Number of samples")
        axes[k].set_ylabel("AUC Score")
        axes[k].set_ylim([np.round(np.min(test_data),1)-0.1,1])
        axes[k].grid()
        
        axes[k].plot(
        x, train_data, ".-", color="r", label="Training score"
        )
        axes[k].plot(
            x, test_data, ".-", color="g", label="Cross-validation score"
        )
        
        if fname == "Synthetic":
            axes[k].vlines(len(target),0,1)
        
        k += 1
        
    fig.tight_layout()
    plt.savefig(fname+'.png',bbox_inches='tight')
    plt.savefig(fname+'.pdf',bbox_inches='tight')
    
        