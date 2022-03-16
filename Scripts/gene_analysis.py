# libraries
import pandas as pd
import numpy as np
import sys
import pickle

from collections import Counter
from pathlib import Path

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
    folder = ""
    
    # Load AMR profile:
    antibiotic_df = pd.read_csv(folder+"/"+name_dataset+'_AMR_data_RSI.csv', header = [0])

    # Load metadata file
    metadata_df = pd.read_csv(folder+"/"+name_dataset+'_metadata.csv', header = [0])

    delimiter = ' '

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
        
        target = target[idx]
        target_str = target_str[idx]
        samples_name = samples[idx]
        
        # Skip antibiotic if the number of samples is too small
        count_class = Counter(target)
        print(count_class)

        if count_class[0] < 12 or count_class[1] < 12:
            continue

        # Load data file
        file_name = folder+"/data_"+method+"_"+name_dataset+"_"+name_antibiotic+'.pickle'
        my_file = Path(file_name)

        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            continue
        else:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)

        # Load features information
        features_df = pd.read_csv(folder+"/"+name_dataset+"_"+name_antibiotic+'_'+method+'_pvalue.csv', header = None)
        n_features = features_df.shape[0]

        print("Number of features = {}".format(n_features))

        features_data = np.array(features_df.loc[:,features_df.columns[0]]).astype(int)

        # Get metadata
        carac = metadata_df.loc[idx,:]
        carac = carac.reset_index(drop=True)
        # Add AR profile column
        carac['AR Profile'] = target_str
        
        # Read each sheet on the BLASTN results
        df_mapping = pd.read_excel(folder+"/"+name_dataset+"_"+method+'_KmersMappingSummaryNP.xlsx', sheet_name=name_antibiotic, header=[0])
        df_mapping = df_mapping.dropna(how='all') 
        df_mapping[df_mapping.columns[0]] = df_mapping[df_mapping.columns[0]].map(lambda x: int(x.lstrip('kmer_').rstrip('.bed')))
        
        kmers_idx = np.array(df_mapping[df_mapping.columns[0]])
        name_genes = np.array(df_mapping["Column7"])

        # Remove all the additional information from gene name
        gene_id = []
        for count, n_genes in enumerate(name_genes):
            if 'Name' in n_genes:
                name_genes[count] = n_genes.split('Name=',1)[1]
                gene_id.append(count)
            elif 'ID=' in n_genes:
                name_genes[count] = n_genes.split('ID=',1)[1]
            elif 'product=' in n_genes:
                name_genes[count] = n_genes.split('product=',1)[1]
                gene_id.append(count)
            elif 'inference' in n_genes:
                name_genes[count] = n_genes.split('ISfinder:',1)[1]
                gene_id.append(count)
            else:
                print(n_genes)

        uni_genes = Counter(name_genes[gene_id])      
        
        # Find the k-mers associated to each gene
        threshold = len(uni_genes) 
        gene_data = np.zeros((len(samples_name),threshold))
        uni_genes_mc = uni_genes.most_common(threshold)
        name_array = []
        count = 0
        for tup in uni_genes_mc:
            n_genes = tup[0]
            if count>threshold-1:
                continue
            freq_genes = uni_genes[n_genes]
            id_n_genes = np.where(name_genes == n_genes)[0]
            name_array.append(n_genes)
            samples_dummy = []
            for k_idx in kmers_idx[id_n_genes]:
                data_idx = data[:,k_idx]
                samples_idx = np.where(data_idx > 0)[0]

                gene_data[samples_idx,count] +=1
            count +=1
        gene_data[gene_data>0]=1 
        name_array = np.array(name_array)

        sample_carac = carac.loc[:,'Sample ID']

        # Find the frequencies of genes for each subset of the data set
        sample_id_C = np.where(carac['Human_Chicken_Environment'] == 'Chicken')[0]
        sample_id_H = np.where(carac['Human_Chicken_Environment'] == 'Human')[0]
        sample_id_E = np.where(carac['Human_Chicken_Environment'] == 'Environment')[0]
        sample_id_SH = np.where(carac['Location'] == 'SH')[0]
        sample_id_FM = np.where(carac['Location'] == 'FM')[0]
        sample_id_R = np.where(carac['AR Profile'] == 'R')[0]
        sample_id_S = np.where(carac['AR Profile'] == 'S')[0]

        gene_analsysis_matrix = np.zeros((12,gene_data.shape[1]))
        for col in range(gene_data.shape[1]):
            #Resistant
            gene_analsysis_matrix[0,col] = 100*np.sum(gene_data[sample_id_R,col])/len(sample_id_R)
            #Sensitive
            gene_analsysis_matrix[1,col] = 100*np.sum(gene_data[sample_id_S,col])/len(sample_id_S)
            #Resistant and Chicken
            id_R_C = np.intersect1d(sample_id_R,sample_id_C)
            if len(id_R_C) == 0:
                gene_analsysis_matrix[2,col] = np.nan
            else:
                gene_analsysis_matrix[2,col] = 100*np.sum(gene_data[id_R_C,col])/len(id_R_C)
            #Resistant and Human
            id_R_H = np.intersect1d(sample_id_R,sample_id_H)
            if len(id_R_H) == 0:
                gene_analsysis_matrix[3,col] = np.nan
            else:
                gene_analsysis_matrix[3,col] = 100*np.sum(gene_data[id_R_H,col])/len(id_R_H)
            #Resistant and Environment
            id_R_E = np.intersect1d(sample_id_R,sample_id_E)
            if len(id_R_E) == 0:
                gene_analsysis_matrix[4,col] = np.nan
            else:
                gene_analsysis_matrix[4,col] = 100*np.sum(gene_data[id_R_E,col])/len(id_R_E)
            #Resistant and Farm
            id_R_FM = np.intersect1d(sample_id_R,sample_id_FM)
            if len(id_R_FM) == 0:
                gene_analsysis_matrix[5,col] = np.nan
            else:
                gene_analsysis_matrix[5,col] = 100*np.sum(gene_data[id_R_FM,col])/len(id_R_FM)
            #Resistant and Slaughterhouse
            id_R_SH = np.intersect1d(sample_id_R,sample_id_SH)
            if len(id_R_SH) == 0:
                gene_analsysis_matrix[6,col] = np.nan
            else:
                gene_analsysis_matrix[6,col] = 100*np.sum(gene_data[id_R_SH,col])/len(id_R_SH)
            #Sensitive and Chicken
            id_S_C = np.intersect1d(sample_id_S,sample_id_C)
            if len(id_S_C) == 0:
                gene_analsysis_matrix[7,col] = np.nan
            else:
                gene_analsysis_matrix[7,col] = 100*np.sum(gene_data[id_S_C,col])/len(id_S_C)
            #Sensitive and Human
            id_S_H = np.intersect1d(sample_id_S,sample_id_H)
            if len(id_S_H) == 0:
                gene_analsysis_matrix[8,col] = np.nan
            else:
                gene_analsysis_matrix[8,col] = 100*np.sum(gene_data[id_S_H,col])/len(id_S_H)
            #Sensitive and Environment
            id_S_E = np.intersect1d(sample_id_S,sample_id_E)
            if len(id_S_E) == 0:
                gene_analsysis_matrix[9,col] = np.nan
            else:
                gene_analsysis_matrix[9,col] = 100*np.sum(gene_data[id_S_E,col])/len(id_S_E)
            #Sensitive and Farm
            id_S_FM = np.intersect1d(sample_id_S,sample_id_FM)
            if len(id_S_FM) == 0:
                gene_analsysis_matrix[10,col] = np.nan
            else:
                gene_analsysis_matrix[10,col] = 100*np.sum(gene_data[id_S_FM,col])/len(id_S_FM)
            #Sensitive and Slaughterhouse
            id_S_SH = np.intersect1d(sample_id_S,sample_id_SH)
            if len(id_S_SH) == 0:
                gene_analsysis_matrix[11,col] = np.nan
            else:
                gene_analsysis_matrix[11,col] = 100*np.sum(gene_data[id_S_SH,col])/len(id_S_SH)

        # Save the results
        index_name = ['Resistant - '+str(len(sample_id_R))+' samples',
            'Sensitive - '+str(len(sample_id_S))+' samples',
            'Resistant and Chicken - ' +str(len(id_R_C))+' samples',
            'Resistant and Human - '+str(len(id_R_H))+' samples',
            'Resistant and Environment - '+str(len(id_R_E))+' samples',
            'Resistant and Farm - '+str(len(id_R_FM))+' samples',
            'Resistant and Slaughterhouse - '+str(len(id_R_SH))+' samples',
            'Sensitive and Chicken - '+str(len(id_S_C))+' samples',
            'Sensitive and Human - '+str(len(id_S_H))+' samples',
            'Sensitive and Environment - '+str(len(id_S_E))+' samples',
            'Sensitive and Farm - '+str(len(id_S_FM))+' samples',
            'Sensitive and Slaughterhouse - '+str(len(id_S_SH))+' samples']
        
        df_gene = pd.DataFrame(data=gene_analsysis_matrix,index=index_name,columns=name_array)
        df_gene.to_csv(folder+'/'+method+'/AMR_Genes_analysis_'+method+'_'+name_dataset+'_'+name_antibiotic+'.csv')
        
        
        ind_inherited = []
        ind_transmission = []
        for col in range(gene_analsysis_matrix.shape[1]):
            if abs(gene_analsysis_matrix[0,col]-gene_analsysis_matrix[1,col]) < 5:
                ind_inherited.append(col)

            
            if abs(gene_analsysis_matrix[0,col]-gene_analsysis_matrix[1,col]) > 30:
                ind_transmission.append(col)

        
        if len(ind_inherited) > 0:
            ind_inherited = np.array(ind_inherited).astype(int)
            df_gene_inherited = pd.DataFrame(data=gene_analsysis_matrix[:,ind_inherited],index=index_name,columns=name_array[ind_inherited])
            df_gene_inherited.to_csv(folder+'/'+method+'/AMR_Genes_Inherited_'+method+'_'+name_dataset+'_'+name_antibiotic+'.csv')

            np.savetxt(folder+'/'+method+'/AMR_Genes_Inherited_'+method+'_'+name_dataset+'_'+name_antibiotic+'.txt', name_array[ind_inherited], fmt='%s')
        
        if len(ind_transmission) > 0:
            ind_transmission = np.array(ind_transmission).astype(int) 
            df_gene_transmission = pd.DataFrame(data=gene_analsysis_matrix[:,ind_transmission],index=index_name,columns=name_array[ind_transmission])
            df_gene_transmission.to_csv(folder+'/'+method+'/AMR_Genes_Acquired_'+method+'_'+name_dataset+'_'+name_antibiotic+'.csv')
            
            np.savetxt(folder+'/'+method+'/AMR_Genes_Acquired_'+method+'_'+name_dataset+'_'+name_antibiotic+'.txt', name_array[ind_transmission], fmt='%s')