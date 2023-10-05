import ast
import numpy as np
import pandas as pd
from pathlib import Path

datasets = ['fBIRNFNCFA', 'fBIRNFNCsMRI', 'fBIRNICAFNC',
            'fBIRNICAFA', 'fBIRNFAsMRI', 'fBIRNICAsMRI']

cmind_clusters = np.zeros((6, len(datasets), 2))
panss_pos_clusters = np.zeros((6, len(datasets), 2))
panss_neg_clusters = np.zeros((6, len(datasets), 2))

sz_clusters = np.zeros((6, len(datasets)))

cmind_clusters[:] = np.nan
panss_pos_clusters[:] = np.nan
panss_neg_clusters[:] = np.nan

folds = list(range(10))
for (d_ix, dataset) in enumerate(datasets):
    dataset_df = pd.read_csv(f'info_df_{dataset.lower()}.csv', index_col=0)
    dataset_df.set_index('idc', inplace=True)
    print(dataset_df['sz'].value_counts())
    assignment_df = pd.read_csv(f'assignments/{dataset.lower()}_assignments.csv', index_col=0)
    assignment_df.columns = assignment_df.columns.map(int)
    clusters = assignment_df.index.values
    for (c_ix, cluster) in enumerate(clusters):
        cmind_folds = np.zeros((len(folds), ))
        panss_pos_folds = np.zeros((len(folds), ))
        panss_neg_folds = np.zeros((len(folds), ))
        sz_folds = np.zeros((len(folds), ))
        for fold in folds:   
            subjects = ast.literal_eval(assignment_df.loc[cluster, fold])
            
            # Both -9999 and nan are nans
            cmind_scores = dataset_df.loc[subjects, 'cminds'].values
            cmind_scores = cmind_scores[~np.isnan(cmind_scores)]
            cmind_scores = cmind_scores[cmind_scores!=-9999.0]

            panss_pos_scores = dataset_df.loc[subjects, 'panss_pos'].values      
            panss_pos_scores = panss_pos_scores[dataset_df.loc[subjects, 'sz'].values]
            panss_pos_scores = panss_pos_scores[~np.isnan(panss_pos_scores)]
            panss_pos_scores = panss_pos_scores[panss_pos_scores != -9999.0]
            
            panss_neg_scores = dataset_df.loc[subjects, 'panss_neg'].values      
            panss_neg_scores = panss_neg_scores[dataset_df.loc[subjects, 'sz'].values]
            panss_neg_scores = panss_neg_scores[~np.isnan(panss_neg_scores)]
            panss_neg_scores = panss_neg_scores[panss_neg_scores != -9999.0]
            
            cmind_folds[fold] = np.mean(cmind_scores)
            panss_pos_folds[fold] = np.mean(panss_pos_scores)
            panss_neg_folds[fold] = np.mean(panss_neg_scores)
            
            sz_folds[fold] = (dataset_df.loc[subjects, 'sz'].values.sum() / len(subjects))

        cmind_clusters[c_ix, d_ix, 0] = np.mean(cmind_folds)
        cmind_clusters[c_ix, d_ix, 1] = np.std(cmind_folds)
        panss_pos_clusters[c_ix, d_ix, 0] = np.mean(panss_pos_folds)
        panss_pos_clusters[c_ix, d_ix, 1] = np.std(panss_pos_folds)
        panss_neg_clusters[c_ix, d_ix, 0] = np.mean(panss_neg_folds)
        panss_neg_clusters[c_ix, d_ix, 1] = np.std(panss_neg_folds)
        sz_clusters[c_ix, d_ix] = np.mean(sz_folds)

dataset_names = ['FAsFNC', 'sMRIsFNC', 'ICAsFNC',
                 'FAICA', 'FAsMRI', 'sMRIICA']

# Generate LaTeX table
table_string = '\hline\n'
cluster_info = [(cmind_clusters, 'CM'), (panss_neg_clusters, 'PN'), (panss_pos_clusters, 'PP')]
for cluster in range(6):
    for (type_data, type_name) in cluster_info:
        table_string += f'{cluster}-{type_name} & \n'
        for (d_ix, dataset_name) in enumerate(dataset_names):
            table_string += f'\cellcolor{"{"}{dataset_name.lower()}c{cluster}{"}"}'
            if np.isnan(type_data[cluster, d_ix, 0]):
                table_string += '- '
            else:
                if np.round(sz_clusters[cluster, d_ix], 2) >= 0.70:
                    table_string += '\\textbf{'
                table_string += f'{np.round(type_data[cluster, d_ix, 0], 1)}+-{np.round(type_data[cluster, d_ix, 1], 1)}'
                if np.round(sz_clusters[cluster, d_ix], 2) >= 0.70:
                    table_string += '}'
                table_string += ' '
            # Last dataset
            if dataset_name == 'sMRIICA':
                table_string += '\\'
                table_string += '\\'
                table_string += '\n'
            else:
                table_string += '&\n'
    table_string += '\hline\n'
print(table_string)

print(sz_clusters)