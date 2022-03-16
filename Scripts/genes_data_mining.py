# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pickle
import colorcet as cc
import seaborn as sns
import community

from pathlib import Path

from adjustText import adjust_text
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from matplotlib.lines import Line2D
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.cluster import hierarchy
from matplotlib import cm


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
    mpl.rc('font',family='Arial')
    method = "model"
    #Add the name of the data set
    name_dataset = ""

    #Add the folder where the results must be saved
    folder = ""
    
    # Load AMR profile:
    antibiotic_df = pd.read_csv(folder+"/"+name_dataset+'_AMR_data_RSI.csv', header = [0])

    # Load metadata file
    metadata_df = pd.read_csv(folder+"/"+name_dataset+'_metadata.csv', header = [0])
    
    # Load known AMR Genes
    known_db = pd.read_csv(folder+"/known genes.csv", header = [0])

    delimiter = ' '

    sample_name_duplicate = []
    for i_name in metadata_df["Sample Name"]:
        if i_name == "PDRFT1S301" or i_name == "PDBST1S301" or i_name == "PDSST1S301" or i_name == "PDSSL101":
            sample_name_duplicate.append("f101")
        elif i_name == "PDRFT1S302" or i_name == "PDRFT2S301" or i_name == "PDRFL103" or i_name == "PDRFL201" \
            or i_name == "PDBST1S302" or i_name == "PDBSL103" or i_name == "PDSST1S302" or i_name == "PDSST2S301":
            sample_name_duplicate.append("f102")
        elif i_name == "PDRFT1S303" or i_name == "PDRFT2S302" or i_name == "PDRFL102" or i_name == "PDBST1S303" \
            or i_name == "PDBSL102":
            sample_name_duplicate.append("f103")
        elif i_name == "PDRFT1S405" or i_name == "PDBST1S405" or i_name == "PDSST1S405":
            sample_name_duplicate.append("f104")
        elif i_name == "PDRFT1S406" or i_name == "PDRFT2S405" or i_name == "PDBST1S406" or i_name == "PDSST1S406":
            sample_name_duplicate.append("f107")
        elif i_name == "PDRFT1S407" or i_name == "PDRFT2S406" or i_name == "PDSST1S407":
            sample_name_duplicate.append("f108")
        elif i_name == "PDRFT1S408" or i_name == "PDRFT207" or i_name == "PDSST207":
            sample_name_duplicate.append("f109")
        elif i_name == "PDRFT109" or i_name == "PDRFYL105" or i_name == "PDRFYL205" or i_name == "PDBST109" \
            or i_name == "PDSSYL205":
            sample_name_duplicate.append("f106")
        else:
            sample_name_duplicate.append('')

    metadata_df["duplicates"] = sample_name_duplicate
    
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
        file_name = "data_"+method+"_"+name_dataset+"_"+name_antibiotic+'.pickle'
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

        features_data = np.array(features_df.loc[:,features_df.columns[0]]).astype(int)
        
        # Include in the metadata dataframe a column with the AR profile
        carac = metadata_df.loc[idx,:]
        carac['AR Profile'] = target_str
        
        # Read each sheet on the BLASTN results
        df_mapping = pd.read_excel(name_dataset+"_"+method+'_KmersMappingSummaryNP.xlsx', sheet_name=name_antibiotic, header=[0])
        df_mapping = df_mapping.dropna(how='all') 
        df_mapping[df_mapping.columns[0]] = df_mapping[df_mapping.columns[0]].map(lambda x: int(x.lstrip('kmer_').rstrip('.bed')))
        
        kmers_idx = np.array(df_mapping[df_mapping.columns[0]])
        name_genes = np.array(df_mapping["Gene Name"])
        sample_mapped = np.array(df_mapping["Sample ID"])

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

        threshold = len(uni_genes) #int(np.ceil(0.1*len(uni_genes)))
        print(threshold)
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
            
            samples_list = sample_mapped[id_n_genes]

            for s_name in samples_list:
                s_idx = np.where(samples_name == s_name)[0]
                gene_data[s_idx,count] += 1
                        
            count +=1
        
        gene_data[gene_data>0]=1 
        name_array = np.array(name_array)

        delta = 30

        id_genes = []
        all_genes = np.array(known_db["Genes"])
        for n, line in enumerate(open('AMR_Genes_Acquired_WGS_'+'_'+str(delta)+'_'+method+'_'+name_dataset+'_'+name_antibiotic+'.txt','r')):
            name_g = np.array(line.split('\n'))[0]
            
            dummy = np.where(name_array == name_g)[0]
            id_genes.append(dummy[0])
            
        if len(id_genes) == 0:
            continue

        gene_data = gene_data[:,id_genes]
        name_array = name_array[id_genes]
        
        id_known = []
        new_name_array = ['' for i in range(len(name_array))]
        for name_g in name_array:
            id_name_g = np.where(all_genes == name_g)[0]
            check_knwon=np.array(known_db.iloc[id_name_g,1])
            if check_knwon > 0:  
                dummy = np.where(name_array == name_g)[0] 
                new_name = np.array(known_db.iloc[id_name_g,2])
                new_name_array[dummy[0]] = new_name[0]
                if check_knwon == 1:   
                    id_known.append(dummy[0])
            else:
                dummy = np.where(name_array == name_g)[0] 
                new_name_array[dummy[0]] = name_g
        
        x_names_uni = np.array(new_name_array)
        
        data_clustermap = gene_data
        print("len id gene = {}".format(len(id_genes)))

        threshold = 10
        neigh = NearestNeighbors(n_neighbors=threshold,metric='euclidean').fit(gene_data)
        mat = neigh.kneighbors_graph(gene_data, mode='distance')
    
        
        adj_matrix = pairwise_distances(gene_data,metric='euclidean')
        adj_matrix[np.tril_indices(len(samples_name), 0)] = np.nan
        adj_df = pd.DataFrame(data=adj_matrix,index=samples_name, columns=samples_name)
        lst = adj_df.stack().reset_index()
        lst = lst.rename(columns={lst.columns[0]:"from", lst.columns[1]:"to", lst.columns[2]:"edge_value"})
        
        # Build your graph
        G = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph(), edge_attribute='edge_value')
        mapping = {}
        for count, s_name in enumerate(samples_name):
            mapping[count] = s_name
        T = nx.relabel_nodes(G, mapping)
        
        partition = community.best_partition(T)
        
        # Get information about the partitions
        pval = []
        pkey = []
        for p_key in partition:
            pval.append(partition[p_key])
            pkey.append(p_key)

        partition_values = []
        pkey = np.array(pkey)
        for i in np.unique(pval):
            id_p = np.where(pval == i)[0]
            partition_values.append(pkey[id_p])

        modularity = community.modularity(partition, T)
        print(nx.algorithms.community.quality.coverage(T,partition_values))
        print(nx.algorithms.community.quality.modularity(T,partition_values))
        print(nx.algorithms.community.quality.performance(T,partition_values))
        
        # The order of the node for networkX is the following order:
        # Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!
            
        sample_carac = carac.loc[:,'Sample ID']

        carac = carac.set_index('Sample ID')
        carac = carac.reindex(T.nodes())        
        
        # And I need to transform my categorical column in a numerical value: group1->1, group2->2...
        carac['Human_Chicken_Environment']=pd.Categorical(carac['Human_Chicken_Environment'])
        my_color_node_Type = carac['Human_Chicken_Environment'].cat.codes

        carac['Collection date']=pd.Categorical(carac['Collection date'])
        my_color_node_Date = carac['Collection date'].cat.codes

        carac['Isolate Source']=pd.Categorical(carac['Isolate Source'])
        my_color_node_Source = carac['Isolate Source'].cat.codes

        lst['edge_value'] = pd.Categorical(lst['edge_value'])
        my_color_edge = lst['edge_value'].cat.codes

        # Define color of the legends        
        ColorLegend_Node_Type = {'Chicken': 0,'Environment': 1,'Human': 2}
        ColorLegend_Node_Date = {'L1': 0,'L2': 1,'L3': 2, 'T1': 3, 'T2': 4, 'T3': 5}
        ColorLegend_Node_Source = {'BS': 0,'DT': 1,'GS': 2, 'JF': 3, 'RF': 4, 'SS': 5, 'MC': 6,'SL': 7,'SX': 8, 'TR': 9, 'TW': 10}
        ColorLegend_Edge = {}
        Legend_Edge = {}
        
        edge_value_array = []
        for edge in T.edges():
            edge_val = T.get_edge_data(edge[0], edge[1])
            edge_value_array.append(edge_val["edge_value"])
        uni_edge_val = np.unique(edge_value_array)

        for count, n_gene in enumerate(uni_edge_val):
            ColorLegend_Edge[n_gene] = count
            Legend_Edge[count] = n_gene

        values_edge = []
        for edge in T.edges():
            edge_val = T.get_edge_data(edge[0], edge[1])
            values_edge.append(ColorLegend_Edge[edge_val['edge_value']])

        values_node_Type = []
        values_node_Date = []
        values_node_Source = []
        for node  in T.nodes():
            values_node_Type.append(ColorLegend_Node_Type[carac.loc[node,'Human_Chicken_Environment']])
            values_node_Date.append(ColorLegend_Node_Date[carac.loc[node,'Collection date']])
            values_node_Source.append(ColorLegend_Node_Source[carac.loc[node,'Isolate Source']])

        values_node_Type = np.array(values_node_Type)
        values_node_Date = np.array(values_node_Date)
        values_node_Source = np.array(values_node_Source)
        
        # compute maximum value s.t. all colors can be normalised
        maxval_node_Type = np.max(values_node_Type) 
        maxval_node_Date = np.max(values_node_Date) 
        maxval_node_Source = np.max(values_node_Source) 
        maxval_edge = np.max(values_edge) 
        
        # get colormap
        cmap_Type=cm.Accent
        cmap_Date=cm.tab10
        cmap_Source=cm.Paired
        cmap_community = cm.tab20

        if len(np.unique(values_edge)) < 21:
            cmap_edge=cm.tab20 #Accent
        else:
            cmap_edge=cm.gist_rainbow

        df = pd.DataFrame(index=T.nodes(), columns=T.nodes())
        for row, data_val in nx.shortest_path_length(T):
            for col, dist in data_val.items():
                df.loc[row,col] = dist

        df = df.fillna(df.max().max())

        pos = nx.kamada_kawai_layout(T, dist=df.to_dict())

        nodes = np.array(T.nodes())
        fig, ax = plt.subplots(nrows = 1, ncols=3, figsize=(30,10))

        ### Subplot 1: Communities and pathogenicity ###

        #Nodes Resistant
        c_values = set(partition.values())
        maxval_community = len(c_values)-1

        partition_vals_int = np.fromiter(partition.values(), dtype=int)
        
        id_commensal = np.where(carac['PathoInfo'] == 'commensal')[0]
        nx.draw_networkx_nodes(T, pos, node_size=100, nodelist= nodes[id_commensal], node_color=[cmap_community(v/maxval_community) for v in partition_vals_int[id_commensal]], 
            cmap = cmap_community, node_shape = 'o', ax=ax[0])
        
        
        id_pathogenic = np.where(carac['PathoInfo'] == 'pathogenic')[0]
        nx.draw_networkx_nodes(T, pos, node_size=100, nodelist= nodes[id_pathogenic], node_color=[cmap_community(v/maxval_community) for v in partition_vals_int[id_pathogenic]], 
            cmap = cmap_community, node_shape = '*', ax=ax[0])
        
        #Edges
        nx.draw_networkx_edges(T,pos, alpha = 0.05,edge_color='k', edge_cmap=cmap_edge, ax=ax[0])
        
        legend_communities = []
        for v in c_values:
            label = "C"+str(v)
            legend_communities.append(Line2D([], [], marker='s', markeredgecolor=cmap_community(v/maxval_community), label=label,
                          color = 'w', markerfacecolor = cmap_community(v/maxval_community), markersize=10))
        legend_communities.append(Line2D([], [], marker='o', markeredgecolor='k', label='Nonpathogenic',
                          color = 'w', markerfacecolor = 'k', markersize=10))
        legend_communities.append(Line2D([], [], marker='*', markeredgecolor='k', label='Pathogenic',
                          color = 'w', markerfacecolor = 'k', markersize=10))
        ax[0].legend(handles = legend_communities, loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=5, fontsize = 13)
        ax[0].set_title("Communities - Modularity: "+str(np.round(nx.algorithms.community.quality.modularity(T,partition_values),4)), fontsize = 13)
        
        ### Subplot 2: ABR and Type ###

        #Nodes Resistant
        id_R = np.where(carac['AR Profile'] == 'R')[0]
        nx.draw_networkx_nodes(T, pos, nodelist= nodes[id_R], node_color=[cmap_Type(v/maxval_node_Type) for v in values_node_Type[id_R]], 
            cmap = cmap_Type, node_size=100, node_shape = 'o', ax=ax[1])
        
        #Nodes Sensitive
        id_S = np.where(carac['AR Profile'] == 'S')[0]
        nx.draw_networkx_nodes(T, pos, nodelist= nodes[id_S], node_color=[cmap_Type(v/maxval_node_Type) for v in values_node_Type[id_S]], 
            cmap = cmap_Type, node_size=100, node_shape = '*', ax=ax[1])

        #Edges
        nx.draw_networkx_edges(T,pos, alpha = 0.3,edge_color=[cmap_edge(v/maxval_edge) for v in values_edge], edge_cmap=cmap_edge, ax=ax[1])
        
        dict_nodes = {}
        for node_name in nodes[id_S]:
            dict_nodes[node_name] = node_name

        legend_elements = []
        for v in set(values_node_Type):
            if v == 0:
                label = "Chicken"
            elif v == 1:
                label = "Environment"
            else:
                label = "Human"
            legend_elements.append(Line2D([], [], marker='s', markeredgecolor=cmap_Type(v/maxval_node_Type), label=label,
                          color = 'w', markerfacecolor = cmap_Type(v/maxval_node_Type), markersize=10))
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='k', label='Resistant',
                          color = 'w', markerfacecolor = 'k', markersize=10))
        legend_elements.append(Line2D([], [], marker='*', markeredgecolor='k', label='Susceptible',
                          color = 'w', markerfacecolor = 'k', markersize=10))
        for v in set(values_edge):
            label = np.round(Legend_Edge[v],4)
            legend_elements.append(Line2D([0], [0], color=cmap_edge(v/maxval_edge), lw=4, label=label))
        ax[1].legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=4, fontsize = 13)
        ax[1].set_title("ABR and Type", fontsize = 13)

        ### Subplot 3: Source and Location ###

        #Nodes Farm
        id_FM = np.where(carac['Location'] == 'FM')[0]
        nx.draw_networkx_nodes(T, pos, nodelist= nodes[id_FM], node_color=[cmap_Source(v/maxval_node_Source) for v in values_node_Source[id_FM]], 
            cmap = cmap_Source, node_size=100, node_shape = 'o', ax=ax[2], alpha=1)
        
        #Nodes Slaughterhouse
        id_SH = np.where(carac['Location'] == 'SH')[0]
        nx.draw_networkx_nodes(T, pos, nodelist= nodes[id_SH], node_color=[cmap_Source(v/maxval_node_Source) for v in values_node_Source[id_SH]], 
            cmap = cmap_Source, node_size=100, node_shape = '*', ax=ax[2], alpha=1)

        #Edges
        nx.draw_networkx_edges(T,pos, alpha = 0.05, edge_color='k', edge_cmap=cmap_edge, ax=ax[2])
        
        legend_elements = []
        for v in set(values_node_Source):
            if v == 0:
                label = "Nose swab"
            elif v == 1:
                label = "Chicken carcass"
            elif v == 2:
                label = "Anal swab"
            elif v == 3:
                label = "Chicken faeces"
            elif v == 4:
                label = "Human faeces"
            elif v == 5:
                label = "Hand swab"
            elif v == 6:
                label = "Chicken cecal"
            elif v == 7:
                label = "Feeds"
            elif v == 8:
                label = "Water (FM)"
            elif v == 9:
                label = "Soil"
            elif v == 10:
                label = "Water (SH)"
            legend_elements.append(Line2D([], [], marker='s', markeredgecolor=cmap_Source(v/maxval_node_Source), label=label,
                          color = 'w', markerfacecolor = cmap_Source(v/maxval_node_Source), markersize=10))
        
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='k', label='Farm',
                          color = 'w', markerfacecolor = 'k', markersize=10))
        legend_elements.append(Line2D([], [], marker='*', markeredgecolor='k', label='Slaughterhouse',
                          color = 'w', markerfacecolor = 'k', markersize=10))
        
        ax[2].legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=4, fontsize = 13)
        ax[2].set_title("Location and Source", fontsize = 13)

        plt.savefig(name_antibiotic+'_Graph_Genes_transmission_'+method+'_'+name_dataset+'_kamada_kawai_layout.pdf', bbox_inches='tight')
        plt.savefig(name_antibiotic+'_Graph_Genes_transmission_'+method+'_'+name_dataset+'_kamada_kawai_layout.svg', bbox_inches='tight')
        
        ### Build Clustermap ###
        df_meta = metadata_df.loc[idx,:]
        
        for partition_key in partition:
            Index_label = df_meta[df_meta[df_meta.columns[0]] == partition_key].index.tolist() 
            df_meta.loc[Index_label,"Community"] = partition[partition_key]    

        # Get information from metadata dataframe
        target_unique = np.unique(target)
        sample_classes_unique = np.array(["Resistant", "Susceptible"])
        sample_id = np.array(df_meta[df_meta.columns[0]])
        sample_group = np.array(df_meta[df_meta.columns[7]])
        id_FM = np.where(sample_group == "FM")[0]
        id_SH = np.where(sample_group == "SH")[0]
        sample_group[id_FM] = "Farm"
        sample_group[id_SH] = "Slaughterhouse"
        sample_group_unique = np.unique(sample_group)
        sample_source = np.array(df_meta[df_meta.columns[9]])
        sample_source_unique = np.unique(sample_source)
        sample_type = np.array(df_meta[df_meta.columns[5]])
        
        id_BS = np.where(sample_type == "BS")[0]
        id_DT = np.where(sample_type == "DT")[0]
        id_GS = np.where(sample_type == "GS")[0]
        id_JB = np.where(sample_type == "JB")[0]
        id_JF = np.where(sample_type == "JF")[0]
        id_JR = np.where(sample_type == "JR")[0]
        id_JS = np.where(sample_type == "JS")[0]
        id_MC = np.where(sample_type == "MC")[0]
        id_RF = np.where(sample_type == "RF")[0]
        id_SS = np.where(sample_type == "SS")[0]
        id_SL = np.where(sample_type == "SL")[0]
        id_SX = np.where(sample_type == "SX")[0]
        id_TF = np.where(sample_type == "TF")[0]
        id_TH = np.where(sample_type == "TH")[0]
        id_TR = np.where(sample_type == "TR")[0]
        id_TS = np.where(sample_type == "TS")[0]
        id_TW = np.where(sample_type == "TW")[0]

        sample_type[id_BS] = "Nose swab"
        sample_type[id_DT] = "Chicken carcass"
        sample_type[id_GS] = "Anal swab"
        sample_type[id_JB] = "Nose swab"
        sample_type[id_JF] = "Chicken faeces"
        sample_type[id_JR] = "Human faeces"
        sample_type[id_JS] = "Hand swab"
        sample_type[id_MC] = "Chicken cecal"
        sample_type[id_RF] = "Human faeces"
        sample_type[id_SL] = "Feeds"
        sample_type[id_SS] = "Hand swab"
        sample_type[id_SX] = "Water (FM)"
        sample_type[id_TF] = "Human faeces"
        sample_type[id_TH] = "Hand swab"
        sample_type[id_TR] = "Soil"
        sample_type[id_TS] = "Nose swab"
        sample_type[id_TW] = "Water (SH)"

        sample_type_unique = np.unique(sample_type)
        ST_type = np.array(df_meta[df_meta.columns[13]])
        ST_type_unique = np.unique(ST_type)
        Phylo_type = np.array(df_meta[df_meta.columns[14]])
        Phylo_type_unique = np.unique(Phylo_type)
        community_type = np.array(df_meta[df_meta.columns[17]])
        community_type_unique = np.unique(community_type)

        # Set colormaps
        cmap_class = cm.get_cmap('Paired',len(target_unique))
        colormap_class = []
        for it in range(len(target_unique)):
            colormap_class.append(cmap_class(it))

        cmap_group = cm.get_cmap('Set1',2)
        colormap_group = []
        for it in range(len(sample_group_unique)):
            colormap_group.append(cmap_group(it))

        cmap_source = cm.get_cmap('Accent',3)
        colormap_source = []
        for it in range(len(sample_source_unique)):
            colormap_source.append(cmap_source(it))

        cmap_Type=cm.Paired
        colormap_type = []
        for label_n in sample_type_unique:
            if label_n == "Nose swab":
                v = 0
            elif label_n == "Chicken carcass":
                v = 1                
            elif label_n == "Anal swab":
                v = 2
            elif label_n == "Chicken faeces":
                v = 3                
            elif label_n == "Human faeces":
                v = 4                
            elif label_n == "Hand swab":
                v = 5              
            elif label_n == "Chicken cecal": 
                v = 6                
            elif label_n == "Feeds": 
                v = 7                
            elif label_n == "Water (FM)":
                v = 8                
            elif label_n == "Soil":
                v = 9                
            elif label_n == "Water (SH)":
                v = 10
            colormap_type.append(cmap_Type(v/10))

        cmap_ST = cc.cm["glasbey"]
        colormap_ST = []
        for it in range(len(ST_type_unique)):
            colormap_ST.append(cmap_ST(it))

        cmap_Phylo = cm.get_cmap('Accent',8)
        colormap_Phylo = []
        for it in range(len(Phylo_type_unique)):
            colormap_Phylo.append(cmap_Phylo(it))

        cmap_community = cm.tab20
        colormap_community = []
        for it in range(len(community_type_unique)):
            colormap_community.append(cmap_community(it/np.max(community_type_unique)))

        df_class = pd.DataFrame({'class': target_str})
        df_class['class'] = pd.Categorical(df_class['class'])
        my_color_class = df_class['class'].cat.codes
        lut_class = dict(zip(set(target), colormap_class))
        row_colors_class = my_color_class.map(lut_class)
        
        df_group = pd.DataFrame({'Group': sample_group})
        df_group['Group'] = pd.Categorical(df_group['Group'])
        my_color_group = df_group['Group'].cat.codes
        lut_group = dict(zip(np.arange(len(sample_group_unique)), colormap_group))
        row_colors_group = my_color_group.map(lut_group)

        df_source = pd.DataFrame({'Source': sample_source})
        df_source['Source'] = pd.Categorical(df_source['Source'])
        my_color_source = df_source['Source'].cat.codes
        lut_source = dict(zip(np.arange(len(sample_source_unique)), colormap_source))
        row_colors_source = my_color_source.map(lut_source)

        df_type = pd.DataFrame({'type': sample_type})
        df_type['type'] = pd.Categorical(df_type['type'])
        my_color_type = df_type['type'].cat.codes
        lut_type = dict(zip(np.arange(len(sample_type_unique)), colormap_type))
        row_colors_type = my_color_type.map(lut_type)

        df_ST = pd.DataFrame({'ST': ST_type})
        df_ST['ST'] = pd.Categorical(df_ST['ST'])
        my_color_ST = df_ST['ST'].cat.codes
        lut_ST = dict(zip(np.arange(len(ST_type_unique)), colormap_ST))
        row_colors_ST = my_color_ST.map(lut_ST)

        df_Phylo = pd.DataFrame({'Phylo': Phylo_type})
        df_Phylo['Phylo'] = pd.Categorical(df_Phylo['Phylo'])
        my_color_Phylo = df_Phylo['Phylo'].cat.codes
        lut_Phylo = dict(zip(np.arange(len(Phylo_type_unique)), colormap_Phylo))
        row_colors_Phylo = my_color_Phylo.map(lut_Phylo)

        df_community = pd.DataFrame({'community': community_type})
        df_community['community'] = pd.Categorical(df_community['community'])
        my_color_community = df_community['community'].cat.codes
        lut_community = dict(zip(np.arange(len(community_type_unique)), colormap_community))
        row_colors_community = my_color_community.map(lut_community)

        # Plot clustermap
        sns_plot = sns.clustermap(data_clustermap, cmap='Greys', vmin=0, vmax=np.amax(data_clustermap),
            cbar_kws={"shrink": .5}, row_colors=[row_colors_class, row_colors_group, row_colors_source, row_colors_type, row_colors_ST, row_colors_Phylo, row_colors_community], 
            xticklabels=x_names_uni, yticklabels=False, linecolor='black') #
        sns_plot.ax_heatmap.set_xticklabels(sns_plot.ax_heatmap.get_xmajorticklabels(), fontsize = 12, fontname="Arial")
        
        for tick_label in sns_plot.ax_heatmap.axes.get_xticklabels():
            tick_text = tick_label.get_text()
            id_n = np.where(x_names_uni == tick_text)[0]
            
            if id_n in id_known:
                tick_label.set_color('r')
        
        row_linkage = hierarchy.linkage(
            distance.pdist(data_clustermap), method='average')

        col_linkage = hierarchy.linkage(
            distance.pdist(data_clustermap.T), method='average') 

        # Set legends
        l1_patch = []
        for count, label in enumerate(sample_classes_unique):
            l1_patch.append(mpatches.Patch(color=lut_class[count], label=label))

        l2_patch = []
        for count, label in enumerate(sample_group_unique):
            l2_patch.append(mpatches.Patch(color=lut_group[count], label=label))

        l3_patch = []
        for count, label in enumerate(sample_source_unique):
            l3_patch.append(mpatches.Patch(color=lut_source[count], label=label))

        l4_patch = []
        for count, label in enumerate(sample_type_unique):
            l4_patch.append(mpatches.Patch(color=lut_type[count], label=label))

        l8_patch = []
        for count, label in enumerate(Phylo_type_unique):
            l8_patch.append(mpatches.Patch(color=lut_Phylo[count], label=label))

        l9_patch = []
        for count, label in enumerate(community_type_unique):
            label_name = "C" + str(int(label))
            l9_patch.append(mpatches.Patch(color=lut_community[count], label=label_name))

        l1 = sns_plot.ax_col_dendrogram.legend(handles = l1_patch,title='Class', loc="center", ncol=1, bbox_to_anchor=(-0.7, 1.5), prop={'size': 10, 'family': "Arial"})
        l1.get_title().set_fontsize('11')
        l1.get_title().set_fontname('Arial')
        sns_plot.ax_col_dendrogram.add_artist(l1)

        l2 = sns_plot.ax_col_dendrogram.legend(handles = l2_patch,title='Location', loc="center", ncol=1, bbox_to_anchor=(-0.45, 1.5), prop={'size': 10, 'family': "Arial"})
        l2.get_title().set_fontsize('11')
        l2.get_title().set_fontname('Arial')
        sns_plot.ax_col_dendrogram.add_artist(l2)

        l3 = sns_plot.ax_col_dendrogram.legend(handles = l3_patch, title='Source', loc="center", ncol = 1, bbox_to_anchor=(-0.2, 1.5), prop={'size': 10, 'family': "Arial"})
        l3.get_title().set_fontsize('11')
        l3.get_title().set_fontname('Arial')
        sns_plot.ax_col_dendrogram.add_artist(l3)

        l4 = sns_plot.ax_col_dendrogram.legend(handles = l4_patch,title='Type', loc="center",ncol=2, bbox_to_anchor=(0.2, 1.5), prop={'size': 10, 'family': "Arial"})
        l4.get_title().set_fontsize('11')
        l4.get_title().set_fontname('Arial')
        sns_plot.ax_col_dendrogram.add_artist(l4)

        l8 = sns_plot.ax_col_dendrogram.legend(handles = l8_patch,title='Phylogroup', loc="center",ncol=2, bbox_to_anchor=(0.65, 1.5), prop={'size': 10, 'family': "Arial"})
        l8.get_title().set_fontsize('11')
        l8.get_title().set_fontname('Arial')
        sns_plot.ax_col_dendrogram.add_artist(l8)

        l9 = sns_plot.ax_col_dendrogram.legend(handles = l9_patch,title='Community', loc="center",ncol=2, bbox_to_anchor=(0.95, 1.5), prop={'size': 10, 'family': "Arial"})
        l9.get_title().set_fontsize('11')
        l9.get_title().set_fontname('Arial')

        sns_plot.cax.set_visible(False)
        
        sns_plot.savefig(name_antibiotic+'_Clustermap_genes_'+method+'_'+name_dataset+'.pdf', bbox_inches='tight')
        sns_plot.savefig(name_antibiotic+'_Clustermap_genes_'+method+'_'+name_dataset+'.svg', bbox_inches='tight')