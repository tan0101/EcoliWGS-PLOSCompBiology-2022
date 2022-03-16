# libraries
from networkx.algorithms.graphical import is_valid_degree_sequence_havel_hakimi
from networkx.generators.joint_degree_seq import is_valid_joint_degree
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

from matplotlib.lines import Line2D
from sklearn.metrics import pairwise_distances
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

    delimiter = ' '
   
    # Load SNP data
    snp_data = pd.read_csv(folder+"/"+name_dataset+"_CoreSNPs.csv",header=[0])
    
    
    samples_name = np.array(snp_data.columns[1:])
    snp_matrix = np.array(snp_data)
    snp_matrix = np.transpose(snp_matrix[:,1:])

    adj_matrix = pairwise_distances(snp_matrix,metric='hamming')
    adj_matrix[np.tril_indices(len(samples_name), 0)] = np.nan
    distribution_adj = adj_matrix.flatten()
    mask = np.where(~np.isnan(distribution_adj))[0]
    distribution_adj = 133631*distribution_adj[mask]
    
    # Print Mean
    print(np.mean(distribution_adj))
    
    # Print Maximum Value
    print(np.amax(distribution_adj))
    
    # Print Median
    print(np.median(distribution_adj))
    
    # First quartile (Q1) 
    Q1 = np.percentile(distribution_adj, 25, interpolation = 'midpoint') 
    print(Q1)
    
    # Third quartile (Q3) 
    Q3 = np.percentile(distribution_adj, 75, interpolation = 'midpoint') 
    print(Q3)
    
    # Interquaritle range (IQR) 
    IQR = Q3 - Q1 
    
    print(IQR) 
    
    # Plot histogram
    n, bins, patches = plt.hist(distribution_adj)
    plt.show()

    # Select the SNPs with distance below 15
    adj_matrix = adj_matrix*133631
    adj_matrix[adj_matrix>15] = np.nan

    adj_df = pd.DataFrame(data=adj_matrix,index=samples_name, columns=samples_name)
    lst = adj_df.stack().reset_index()
    lst = lst.rename(columns={lst.columns[0]:"from", lst.columns[1]:"to", lst.columns[2]:"edge_value"})

    # Build graph
    T=nx.from_pandas_edgelist(df=lst, source='from', target='to', edge_attr='edge_value', create_using=nx.Graph() )
    
    carac = metadata_df
    sample_carac = carac.loc[:,'Sample ID']

    carac = carac.set_index('Sample ID')
    carac = carac.reindex(T.nodes())        
    
    # Transform my categorical column in a numerical value: group1->1, group2->2...
    carac['Human_Chicken_Environment']=pd.Categorical(carac['Human_Chicken_Environment'])
    my_color_node_Type = carac['Human_Chicken_Environment'].cat.codes

    lst['edge_value'] = pd.Categorical(lst['edge_value'])
    my_color_edge = lst['edge_value'].cat.codes

    # Set color legends        
    ColorLegend_Node_Type = {'Chicken': 0,'Environment': 1,'Human': 2}
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
    
    values_node_Type = np.array(values_node_Type)
    
    # compute maximum value s.t. all colors can be normalised
    maxval_node_Type = np.max(values_node_Type) 
    maxval_edge = np.max(values_edge) 
    
    # get colormap
    cmap_Type=cm.Accent
    
    df = pd.DataFrame(index=T.nodes(), columns=T.nodes())
    for row, data_val in nx.shortest_path_length(T):
        for col, dist in data_val.items():
            df.loc[row,col] = dist

    df = df.fillna(df.max().max())
    pos = nx.kamada_kawai_layout(T, dist=df.to_dict())
    
    nodes = np.array(T.nodes())
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(20,20))
    
    # Find commensal isolates
    id_commensal = np.where(carac['PathoInfo'] == 'commensal')[0]
    nx.draw_networkx_nodes(T, pos, node_size=100, nodelist= nodes[id_commensal], node_color=[cmap_Type(v/maxval_node_Type) for v in values_node_Type[id_commensal]], 
        node_shape = 'o', ax=ax)
    
    # Find Pathogenic isolates
    id_pathogenic = np.where(carac['PathoInfo'] == 'pathogenic')[0]
    nx.draw_networkx_nodes(T, pos, node_size=100, nodelist= nodes[id_pathogenic], node_color=[cmap_Type(v/maxval_node_Type) for v in values_node_Type[id_pathogenic]], 
        node_shape = '*', ax=ax)
    
    # Draw Edges
    nx.draw_networkx_edges(T,pos,alpha = 0.5, edge_color=[cmap_edge(v/maxval_edge) for v in values_edge], edge_cmap=cmap_edge, ax=ax)
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
    legend_elements.append(Line2D([], [], marker='o', markeredgecolor='k', label='Nonpathogenic',
                        color = 'w', markerfacecolor = 'k', markersize=10))
    legend_elements.append(Line2D([], [], marker='*', markeredgecolor='k', label='Pathogenic',
                        color = 'w', markerfacecolor = 'k', markersize=10))
    
    for v in set(values_edge):
        label = np.round(Legend_Edge[v],4)
        legend_elements.append(Line2D([0], [0], color=cmap_edge(v/maxval_edge), lw=4, label=label))
    ax.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
        fancybox=True, shadow=True, ncol=5, fontsize = 25)
    
    plt.savefig(folder+'/SNP_network_kamada_kawai_layout.pdf', bbox_inches='tight')
        
        
        
    