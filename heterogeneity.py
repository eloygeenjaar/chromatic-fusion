import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

datasets = ['fBIRNFNCFA', 'fBIRNFNCsMRI', 'fBIRNICAFNC', 'fBIRNICAFA', 'fBIRNFAsMRI', 'fBIRNICAsMRI']

important_clusters = {
    'fBIRNFNCFA': [3],
    'fBIRNFNCsMRI': [0],
    'fBIRNICAFNC': [2],
    'fBIRNICAFA': [1, 4],
    'fBIRNFAsMRI': [2],
    'fBIRNICAsMRI': [4]
}

names = np.array([
    'FA-sFNC MCP-3',
    'sMRI-sFNC MCP-0',
    'ICA-sFNC MCP-2',
    'FA-ICA MCP-1', 'FA-ICA MCP-4',
    'sMRI-FA MCP-2',
    'sMRI-ICA MCP-4'])

dfs = {dataset: None for dataset in datasets}
for dataset in datasets:
    df = pd.read_csv(f'assignments/{dataset.lower()}_assignments.csv', index_col=0)
    df.columns = df.columns.map(int)
    dfs[dataset] = df

sz_accounted_for = np.zeros((10, ))
sz_accounted_for_num = np.zeros((10, ))
for fold in range(10):
    sz_subjects_accounted_fold = set() 
    for dataset in datasets:
        cur_df = dfs[dataset]
        clusters = important_clusters[dataset]
        for cluster in clusters:
            subjects = ast.literal_eval(cur_df.loc[cluster, fold])
            info_df = pd.read_csv(f'info_df_{dataset.lower()}.csv', index_col=0)
            info_df.set_index('idc', inplace=True)
            sz_mask = info_df.loc[subjects, 'sz'].values
            subjects = set(np.asarray(subjects)[sz_mask].tolist())
            sz_subjects_accounted_fold = sz_subjects_accounted_fold.union(subjects)
            
    sz_accounted_for[fold] = len(sz_subjects_accounted_fold) / info_df['sz'].sum()
    sz_accounted_for_num[fold] = len(sz_subjects_accounted_fold)

print(sz_accounted_for.min(), sz_accounted_for.max())
print(f'SZ subjects accounted for: {sz_accounted_for.mean()}+-{sz_accounted_for.std()}')

enriched_arr = np.zeros((7, 7, 10))
perc_arr = np.zeros((7, 7, 10))
i = 0
for dataset_i in datasets:
    # Get the IDCs for a dataset in a fold
    current_df_i = dfs[dataset_i]
    print(dataset_i)
    clusters_i = important_clusters[dataset_i]
    for cluster_i in clusters_i:
        j = 0
        sz_subjects_accounted_fold = set() 
        for dataset_j in datasets:
            # Get the IDCs for a dataset in a fold
            print(dataset_j)
            current_df_j = dfs[dataset_j]
            clusters_j = important_clusters[dataset_j]
            for cluster_j in clusters_j:
                for fold in range(10):
                    # Subjects for modality pair i and modality pair j for one of the clusters
                    subjects_i = ast.literal_eval(current_df_i.loc[cluster_i, fold])
                    subjects_j = ast.literal_eval(current_df_j.loc[cluster_j, fold])
                    df_i = pd.read_csv(f'info_df_{dataset_i.lower()}.csv', index_col=0)
                    df_i.set_index('idc', inplace=True)
                    # Only look at schizophrenia subjects
                    sz_mask_i = df_i.loc[subjects_i, 'sz'].values
                    subjects_i = set(np.asarray(subjects_i)[sz_mask_i].tolist())
                    df_j = pd.read_csv(f'info_df_{dataset_j.lower()}.csv', index_col=0)
                    df_j.set_index('idc', inplace=True)
                    sz_mask_j = df_j.loc[subjects_j, 'sz'].values
                    subjects_j = set(np.asarray(subjects_j)[sz_mask_j].tolist())
                    # Find the subjects only in the row cluster
                    subjects_i_not_j = subjects_i.difference(subjects_j)
                    # Calculate the percentage of subjects in the row cluster
                    # that are not in the column cluster
                    perc_arr[i, j, fold] = len(subjects_i_not_j) / len(subjects_i) * 100
                j += 1
        i += 1


enriched_arr = enriched_arr.mean(-1)
perc_arr = np.round(perc_arr.mean(-1), 0)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(perc_arr, cmap='hot', square=True, xticklabels=names, yticklabels=names, annot=True, ax=ax, fmt='g', cbar_kws={"shrink": 0.8})
plt.tight_layout()
plt.savefig('overlap.png', dpi=400)
