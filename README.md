# EcoliWGS-PLOSCompBiology-2022
"Whole genome sequencing and gene sharing network analysis powered by machine learning identifies antibiotic resistance sharing between animals, humans and environment in livestock farming" by Zixin Peng, Alexandre Maciel-Guerra, Michelle Baker, Xibin Zhang, Yue Hu, Wei Wang, Jia Rong, Jing Zhang, Ning Xue, Paul Barrow, David Renney, Dov Stekel, Paul Williams, Longhai Liu, Junshi Chen, Fengqin Li and Tania Dottorini accepted for publication on PLOS Computational Biology

Any questions should be made to the corresponding author Dr Tania Dottorini (Tania.Dottorini@nottingham.ac.uk)

Three scripts are available:

1. important_kmers.py -> find the most important k-mers based on the pvalue of the Chi-square test and an Extreme Learning Machine for each antibitioc studied.
2. classification_kmers.py -> measures the performance of 10 different classifiers using a nested cross-validation with the dataset acquired from important_kmers.py
3. genes_analysis.py -> post-process of the genes to select the ones with a relative presence higher than 30%
4. genes_data_mining.py -> create an undirect graph and a clustermap with NetworkX and Seaborn Package based on the genes acquired from genes_analysis.py
5. snp_network.py -> create an undirect graph for the SNPs
6. sample_size_analysis_SMOTE.py -> wrapper backward selection to eliminate the samples in order to study if the number of samples is enough to classify the data and check if overfitting happened
7. sample_size_analysis_Synthetic.py -> wrapper backward selection to eliminate the samples in order to study if the number of samples is enough to classify the data and check if overfitting happened. The data is oversamples as a pre-processing step using a SMOTE approach
8. learning_curves.py -> plot the results found by sample_size_analysis_SMOTE.py and sample_size_analysis_Synthetic.py
