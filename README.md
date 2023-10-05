# Chromatic fusion
The code repository that corresponds to the paper: Chromatic fusion: generative multimodal neuroimaging data fusion


The script to train the DMVAE on fBIRN data is called main.py
The following scripts can be run to obtain the more detailed results in the paper
- cluster_selection.py can be used to find the optimal number of clusters for each modality pair
- run_clustering.py should be run first for the clustering results, it creates table 2 and saves the cluster assignments
- visualize_clustering.py can be used to get the colors for each color (and table 2), it also allows the user to create figure 4
- save_cluster_reconstructions.py saves the reconstructions for each cluster
- generate_sfncs_figure5.py generates the sfncs for Figure 5
- crop_results.py crops the results for Figures 5 and 7
- generate_appendix_cognition.py is used to create appendix D
- heterogeneity.py is used to create Figure 6


To generate Figures 5 and 7 after obtaining all the results:
1) Run generate_slices.py in MRICroGL
2) Run generate_slices_shared.py in MRICRoGL
3) Run generate_sfncs_figure5.py
4) Run generate_sfncs_figure7.py
5) Run crop_results.py

