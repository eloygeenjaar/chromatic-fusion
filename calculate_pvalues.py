import ast
import numpy as np
import pandas as pd

import scipy.stats

from pathlib import Path
from statsmodels.stats.multitest import multipletests


def create_latex_table(array):
    s = ""
    for i in range(array.shape[0]):
        row = f'C{i} '
        for j in range(array[i].size):
            row += f'& {bold(array[i, j])} '
        if i != (array.shape[0] - 1):
            row += r'\\' + '\n'
        else:
            row += ' \n'
        s += row
    print(s)


def bold(val):
    if np.isnan(val):
        return 'NA'
    elif val < 0.05:
        if str(val) == "0.0":
            return r"$<$\textbf{0.0005}"
        else:
            return r"\textbf{" + str(val) + "}"
    else:
        return val


num_folds = 10
cluster_selection = {
    'fBIRNFNCsMRI': 3,
    'fBIRNICAsMRI': 6,
    'fBIRNFAsMRI': 4,
    'fBIRNICAFNC': 4,
    'fBIRNICAFA': 6,
    'fBIRNFNCFA': 4
}
embedding_path = Path('embeddings')
datasets = [
    'fBIRNFNCFA', 'fBIRNFNCsMRI', 'fBIRNICAFNC', 'fBIRNICAFA', 'fBIRNFAsMRI', 'fBIRNICAsMRI'
]
embedding_paths = [embedding_path / dataset for dataset in datasets]
for (dataset, embedding_path) in zip(datasets, embedding_paths):

    # Load the dataframe with the paths and target variables
    info_df = pd.read_csv(Path(f'./info_df_{dataset.lower()}.csv'))
    idcs = info_df['idc'].tolist()

    # Create a subjects-by-folds dataframe to store cluster assignments
    # for a subejct across folds
    subject_clusters = pd.DataFrame(np.zeros((len(idcs), 10)),
                                    index=idcs,
                                    columns=list(range(10)))

    # Get the dataframes that describe what subject is assigned to what cluster across folds
    assignment_df = pd.read_csv(f'assignments/{dataset.lower()}_assignments.csv', index_col=0)
    assignment_df.columns = assignment_df.columns.map(int)

    num_clusters = cluster_selection[dataset]

    # We are trying to determine (across folds), what cluster a subject is most often assigned to
    # So first we populate the subject_clusters dataframe:
    for fold in range(10):
        for cluster in range(num_clusters):
            subjects = ast.literal_eval(assignment_df.loc[cluster, fold])
            subject_clusters.loc[subjects, fold] = int(cluster)

    two_cluster_counter = 0
    # Then we assign a cluster to each subject
    subject_assignments = pd.Series(np.zeros((len(idcs), )), index=idcs)
    for idc in idcs:
        num_assignments = subject_clusters.loc[idc].value_counts().values
        clusters = subject_clusters.loc[idc].value_counts().index.tolist()
        # Note that num_assignments goes from high to low
        if len(num_assignments) > 1:
            # If a subject is assigned to two clusters atleast the same number of times
            # Then use the first cluster. (clusters[0])
            if num_assignments[0] == num_assignments[1]:
                two_cluster_counter += 1
                # Assign the cluster in fold 0
                subject_assignments.loc[idc] = int(clusters[0])
            else:
                # Assign the cluster that the subject is most assigned to
                subject_assignments.loc[idc] = int(clusters[0])
        else:
            subject_assignments.loc[idc] = int(clusters[0])

    # Number of subjects that are assigned equally as much to one and another cluster
    # the max we get for this number is 32 for sMRI-ICA
    print(f'Two cluster counter: {two_cluster_counter}')
    p_values = np.zeros((num_clusters, num_clusters))
    info_df = pd.read_csv(Path(f'./info_df_{embedding_path.name.lower()}.csv'), index_col=0)
    info_df.set_index('idc', inplace=True)
    p_vals_sex = np.zeros((num_clusters, num_clusters))
    p_vals_sz = np.zeros((num_clusters, num_clusters))
    for c_1 in range(num_clusters):
        for c_2 in range(num_clusters):
            if c_1 != c_2:
                # Find subjects assigned to cluster c_1
                indices_c1 = subject_assignments.index[subject_assignments == c_1].tolist()
                # Find subject assigned to cluster c_2
                indices_c2 = subject_assignments.index[subject_assignments == c_2].tolist()
                assignments_c1 = np.ones((len(indices_c1), ))
                assignments_c2 = np.zeros((len(indices_c2), ))
                # Check sex and diagnosis
                labels_sz_c1 = info_df.loc[indices_c1, 'sz'].values
                labels_sex_c1 = info_df.loc[indices_c1, 'sex'].values
                labels_sz_c2 = info_df.loc[indices_c2, 'sz'].values
                labels_sex_c2 = info_df.loc[indices_c2, 'sex'].values
                # Create assignments and labels and then correlate
                assignments = np.concatenate((assignments_c1, assignments_c2)).astype(int)

                labels_sz = np.concatenate((labels_sz_c1, labels_sz_c2)).astype(int)
                labels_sex = np.concatenate((labels_sex_c1, labels_sex_c2)).astype(int)
                # Calculate between the clusters whether being assigned to that cluster
                # is correlated with having schizophrenia
                r_val_sz, p_val_sz = scipy.stats.pearsonr(assignments, labels_sz)
                r_val_sz, p_val_sex = scipy.stats.pearsonr(assignments, labels_sex)
                p_vals_sex[c_1, c_2] = p_val_sex
                p_vals_sz[c_1, c_2] = p_val_sz
            else:
                p_vals_sex[c_1, c_2] = np.nan
                p_vals_sz[c_1, c_2] = np.nan

    # Multiple test correction
    for c in range(num_clusters):
        pvals_sex = np.concatenate((p_vals_sex[c, 0:c], p_vals_sex[c, (c+1):]))
        _, pvals_sex, _, _ = multipletests(pvals_sex, method='bonferroni')
        p_vals_sex[c, 0:c] = pvals_sex[:c]
        p_vals_sex[c, (c+1):] = pvals_sex[c:]
        pvals_sz = np.concatenate((p_vals_sz[c, 0:c], p_vals_sz[c, (c+1):]))
        _, pvals_sz, _, _ = multipletests(pvals_sz, method='bonferroni')
        p_vals_sz[c, 0:c] = pvals_sz[:c]
        p_vals_sz[c, (c+1):] = pvals_sz[c:]

    print(embedding_path.name)
    create_latex_table(np.round(p_vals_sex, 3))
    create_latex_table(np.round(p_vals_sz, 3))
