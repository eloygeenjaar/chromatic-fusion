import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import random
import nibabel as nb
from torch import distributions as D
from chromatic.architectures import *
from chromatic.models import DMVAE
from chromatic.datasets import *
from chromatic.runners import DMVAERunner
from torch import distributions as D
from sklearn.manifold import TSNE
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kmeans_jsd import KMeansJSD
from nilearn.image import resample_to_img, resample_img
from scipy import ndimage, optimize
from utils import *


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
template = nb.load('mni152.nii.gz')

priv_size = 16
shared_size = 32
subject_size = 0
beta = 1.0
batch_size = 5
num_features = 8

cluster_selection = {
    'fBIRNFNCsMRI': 3,
    'fBIRNICAsMRI': 6,
    'fBIRNFAsMRI': 4,
    'fBIRNICAFNC': 4,
    'fBIRNICAFA': 6,
    'fBIRNFNCFA': 4
}

model_name = 'DMVAE'
datasets = ['fBIRNICAFA', 'fBIRNFNCsMRI', 'fBIRNFAsMRI', 'fBIRNFNCFA', 'fBIRNICAFNC', 'fBIRNICAsMRI']
for dataset in datasets:

    num_clusters = cluster_selection[dataset]

    # Load the dataframe with the paths and target variablesB
    info_df = pd.read_csv(Path(f'./info_df_{dataset.lower()}.csv'))
    # Load the dataset generator based on the name of the dataset and the dataframe
    dataset_generator = get_dataset_generator(info_df, dataset)
    input_shape = dataset_generator.data_shape

    # Initialize the distributions to a Normal distribution
    dist = D.Normal
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the encoders and decoders based on the dataset
    encoder1, encoder2, decoder1, decoder2 = get_architectures(
        dataset, input_shape, priv_size, shared_size, num_features)
    # Load the model and criterion based on the model
    model, criterion_name = get_model(model_name, input_shape, encoder1, encoder2, decoder1, decoder2, dist, dataset, device)

    # Find the IDCs to initialize the count_df
    init_train_df = dataset_generator.dataframes[0]['train']
    init_valid_df = dataset_generator.dataframes[0]['valid']
    init_test_df = dataset_generator.dataframes[0]['test']
    init_df = pd.concat((init_train_df, init_valid_df, init_test_df), axis=0)

    count_df = pd.DataFrame(np.zeros((len(init_df.index.values), len(init_df.index.values))),
                            index=init_df['idc'], columns=init_df['idc'])

    # Initialize the subject assignments for each cluster, and cluster assignments for each fold
    subj_assignments = pd.DataFrame(np.zeros((len(init_df.index.values), 10)), index=init_df['idc'], columns=list(range(10)))
    fold_assignment_df = pd.DataFrame(np.zeros((num_clusters, 10)), index=list(range(num_clusters)), columns=list(range(10)))
    fold_cluster_assignments = []
    percentages_sz = []
    percentages_sex = []
    percentages_sites = []
    sites_sz = []
    sites_all = []

    total_embeddings = np.zeros((10, len(init_df.index.values), priv_size * 2 + shared_size, 2))
    for fold in range(10):
        mu_df = pd.DataFrame(np.zeros((len(init_df.index.values), priv_size * 2 + shared_size)), index=init_df['idc'], columns=list(range(priv_size * 2 + shared_size)))
        sd_df = pd.DataFrame(np.zeros((len(init_df.index.values), priv_size * 2 + shared_size)), index=init_df['idc'], columns=list(range(priv_size * 2 + shared_size)))
        fold_path = Path(f'logs/{dataset}_{num_features}_{priv_size}_{shared_size}_{beta}_{str(model)}_1e-05_{str(encoder1)}_{str(encoder2)}{str(decoder1)}_{str(decoder2)}_{criterion_name}_6_42_300_new/fold_{fold}')
        if fold_path.is_dir():
            
            # Create loaders
            loaders, pipes = dataset_generator.build_pipes(fold)

            train_loaders = {"train": loaders['train'],
                            "valid": loaders['valid']}
            
            # Load best checkpoint (based on validation performance)
            checkpoint = torch.load(fold_path / Path('best.pth'), map_location=device)
            
            print(f'Fold: {fold}, best epoch: {checkpoint["global_epoch_step"]}')
            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer = criterion = epochs = callbacks = logdir = scheduler = None

            # Initialize runner, but only to call evaluate_loader
            runner = get_runner(model_name, model, train_loaders, optimizer,
                criterion, epochs, callbacks,
                scheduler, logdir, device)

            # Obtain the representations for the training, validation, and test set
            train_tuple, train_targets, tr_dist_tuple = runner.evaluate_loader(
                            loader=loaders['train'])
            valid_tuple, valid_targets, va_dist_tuple = runner.evaluate_loader(
                            loader=loaders['valid'])
            test_tuple, test_targets, te_dist_tuple = runner.evaluate_loader(
                            loader=loaders["test"])

            # Combine private-1, private-2, and shared into a single embedding for the training,
            # validation, and test set
            x_train = np.concatenate((train_tuple[0].cpu().numpy(), train_tuple[1].cpu().numpy(), train_tuple[2].cpu().numpy()), axis=1)
            x_valid = np.concatenate((valid_tuple[0].cpu().numpy(), valid_tuple[1].cpu().numpy(), valid_tuple[2].cpu().numpy()), axis=1)
            train_embs = np.concatenate((x_train, x_valid), axis=0)
            x_test = np.concatenate((test_tuple[0].cpu().numpy(), test_tuple[1].cpu().numpy(), test_tuple[2].cpu().numpy()), axis=1)
            x_all = np.concatenate((x_train, x_valid, x_test), axis=0)
            tr_dist = torch.cat((tr_dist_tuple), dim=-1)
            va_dist = torch.cat((va_dist_tuple), dim=-1)
            te_dist = torch.cat((te_dist_tuple), dim=-1)
            total_dists = D.Normal(
                torch.cat((tr_dist[:, 0], va_dist[:, 0], te_dist[:, 0]), dim=0),
                torch.cat((tr_dist[:, 1], va_dist[:, 1], te_dist[:, 1]), dim=0),
            )

            total_mean = total_dists.mean.cpu().numpy()

            # Load all the targets
            y_train = train_targets.cpu().numpy()
            y_valid = valid_targets.cpu().numpy()
            train_targets = np.concatenate((y_train, y_valid), axis=0)
            y_test = test_targets.cpu().numpy()
            y_all = np.concatenate((y_train, y_valid, y_test), axis=0)

            # Perform the clustering
            #km = KMeansJSD(num_clusters=num_clusters, device=device,
            #               tol=1E-4, min_steps=10)
            #km.fit(total_dists)
            km = KMeans(n_clusters=num_clusters, n_init='auto', max_iter=1000, random_state=42)
            km.fit(total_mean)
            assignments = km.predict(total_mean)

            # Obtain assignments
            print(f'Fold: {fold}, {len(np.unique(assignments))}, {num_clusters}')
            train_df = dataset_generator.dataframes[fold]['train']
            valid_df = dataset_generator.dataframes[fold]['valid']
            test_df = dataset_generator.dataframes[fold]['test']
            fold_df = pd.concat((train_df, valid_df, test_df), axis=0)
            
            # Aggregate embeddings
            mu_df.loc[fold_df['idc'], :] = total_dists.mean.cpu().numpy()
            sd_df.loc[fold_df['idc'], :] = total_dists.stddev.cpu().numpy()
            total_embeddings[fold, :, :, 0] = mu_df.values
            total_embeddings[fold, :, :, 1] = sd_df.values
            subj_assignments.loc[fold_df['idc'], fold] = assignments.copy()

            cluster_assignments = []
            perc_szs, perc_sexs, perc_sites = [], [], []

            # Number of clusters by number of sites
            init_sites_df = pd.DataFrame(np.zeros((num_clusters, 7)), columns=sorted(fold_df['site'].unique().tolist()))
            count_sites_df = pd.DataFrame(np.zeros((num_clusters, 7)), columns=sorted(fold_df['site'].unique().tolist()))
            for i in range(num_clusters):
                # For every subject in this cluster add 1 count
                count_df.loc[fold_df.loc[assignments == i, "idc"].tolist(),
                            fold_df.loc[assignments == i, "idc"].tolist()] += 1
                fold_assignment_df.loc[i, fold] = str(fold_df.loc[assignments == i, "idc"].tolist())
                cluster_assignments.append(fold_df.loc[assignments == i, "idc"].tolist())
                perc_sz = (assignments[fold_df['sz']==1] == i).sum() / (assignments == i).sum()
                perc_sex = (assignments[fold_df['sex']==1] == i).sum() / (assignments == i).sum()

                # Most common site in this cluster divided by total number of assignments
                if (assignments == i).sum() > 0:
                    perc_site = fold_df.loc[assignments == i, 'site'].value_counts().values[0] / (assignments == i).sum()
                    perc_sites.append(np.round(perc_site, 2))
                else:
                    perc_sites.append(0.0)

                # The sites available in this cluster
                sites = fold_df.loc[assignments == i, 'site'].unique().tolist()
                count_sites = fold_df.loc[assignments == i, 'site'].value_counts()

                # How often someone in this cluster is a patient and from a certain site divided by the total number of subjects
                # from that site
                for site in sites:
                    init_sites_df.loc[i, site] = ((assignments[y_all==1] == i) & (fold_df.loc[y_all==1, 'site'] == site)).sum() / count_sites[site]
                    count_sites_df.loc[i, site] = count_sites[site]
                perc_szs.append(np.round(perc_sz, 2))
                perc_sexs.append(np.round(perc_sex, 2))

            percentages_sz.append(perc_szs)
            percentages_sex.append(perc_sexs)
            fold_cluster_assignments.append(cluster_assignments)
            percentages_sites.append(perc_sites)
            sites_sz.append(init_sites_df)
            sites_all.append(count_sites_df)

            if fold == 0:
                clusters = np.zeros((num_clusters, priv_size * 2 + shared_size))
                for temp_c in range(num_clusters):
                    clusters[temp_c] = km.cluster_centers_[temp_c]
                np.save(fold_path / 'clusters.npy', clusters)
                np.save(fold_path / 'assignments.npy', assignments)

    # Calculate the overlap between folds
    overlap_folds = np.zeros((10, num_clusters, 10, num_clusters))
    # Loop over all folds
    for i in range(10):
        # And all clusters
        for j in range(num_clusters):
            # Obtain the set of subjects assigned to cluster j in fold i
            cluster_set = set(fold_cluster_assignments[i][j])
            # Loop over all folds and match the clusters
            for k in range(10):
                # For all other folds check overlap with fold i cluster j
                if k != i:
                    # For all other clusters in these other folds
                    for e in range(num_clusters):
                        # Check if the cluster set is larger than zero
                        if len(cluster_set) != 0:
                            # Obtain cluster set for fold k and cluster e
                            cur_cluster_set = set(fold_cluster_assignments[k][e])
                            # The overlap in fold k, cluster e is the intersection divided by the size of the original cluster set
                            overlap_folds[i, j, k, e] = len(cluster_set.intersection(cur_cluster_set)) / len(cluster_set)
                        else:
                            overlap_folds[i, j, k, e] = 0

    # Map each fold and cluster to fold 0
    mapping_to_fold_0 = np.zeros((num_clusters, 10))
    row_ixs = []
    col_ixs = []
    sz_scores, sex_scores = [percentages_sz[0]], [percentages_sex[0]]
    total_sz_sites = np.zeros((10, num_clusters, 7))
    total_sz_sites[0] = sites_sz[0].values.copy()
    total_sites = sites_all[0].copy()
    mapping_to_fold_0[:, 0] = list(range(num_clusters))
    reassigned_clusters = pd.DataFrame(np.zeros((num_clusters, 10)), index=list(range(num_clusters)), columns=list(range(10)))
    # Reassign the assignments to the matched clusters for fold 0
    for c in range(num_clusters):
        reassigned_clusters.loc[c, 0] = fold_assignment_df.loc[c, 0]
    # For each fold (not zero)
    for i in range(1, 10):
        # Find the mapping with the most average overlap between clusters
        # using the Hungarian algorithm
        row_ix, col_ix = optimize.linear_sum_assignment(-overlap_folds[0, :, i, :])
        # col_ix will be how each cluster in fold i maps to fold 0
        mapping_to_fold_0[:, i] = col_ix
        # Obtain the percentages of schizophrenia patients for fold i
        # and the cluster that corresponds to cluster c in fold 0
        temp_sz, temp_sex = [], []
        for c in range(num_clusters):
            temp_sz.append(percentages_sz[i][col_ix[c]])
            temp_sex.append(percentages_sex[i][col_ix[c]])
            reassigned_clusters.loc[c, i] = fold_assignment_df.loc[col_ix[c], i]
        # 
        total_sz_sites[i] = sites_sz[i].iloc[col_ix, :].values
        total_sites += sites_all[i].iloc[col_ix, :].values
        sz_scores.append(temp_sz)
        sex_scores.append(temp_sex)
        row_ixs.append(row_ix)
        col_ixs.append(col_ix)
    med_sites_df = np.median(total_sz_sites, axis=0)
    avg_counts = total_sites / 10
    avg_cost = np.zeros((9, num_clusters))
    for i in range(1, 10):
        print(overlap_folds[0, row_ixs[i-1], i, col_ixs[i-1]])
        print(row_ixs[i-1])
        avg_cost[i-1, row_ixs[i-1]] = overlap_folds[0, row_ixs[i-1], i, col_ixs[i-1]]

    sz_scores_m = np.mean(np.asarray(sz_scores), axis=0)
    print(f'--- Dataset: {dataset} ---')
    print('Scores:')
    print('R')
    print(np.round(np.mean(avg_cost, axis=0), 2))
    print(np.round(np.std(avg_cost, axis=0), 2))
    print('SZ')
    print(np.round(np.mean(np.asarray(sz_scores), axis=0), 2))
    print(np.round(np.std(np.asarray(sz_scores), axis=0), 2))
    print('Sex')
    print(np.round(np.mean(np.asarray(sex_scores), axis=0), 2))
    print(np.round(np.std(np.asarray(sex_scores), axis=0), 2))

    # Save all embeddings data
    embedding_path = Path(f'embeddings/{dataset}')
    if not embedding_path.is_dir():
        embedding_path.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path / 'embeddings.npy', total_embeddings)
    np.save(embedding_path / 'cluster_mapping_folds.npy', mapping_to_fold_0)
    subj_assignments.to_csv(embedding_path / 'assignment.csv')
    print('--- Embeddings saved ---')


    print('Site standard deviations')
    sd_sites = np.zeros((num_clusters,))
    for i in range(num_clusters):
        sd_sites[i] = (((med_sites_df[i] - sz_scores_m[i]) ** 2) * avg_counts.loc[i].values).sum() / avg_counts.loc[i].sum()
    print(np.round(sd_sites, 2))
    print(avg_counts.round(2).to_markdown())

    reassigned_clusters.to_csv(f'assignments/{dataset.lower()}_assignments.csv')
