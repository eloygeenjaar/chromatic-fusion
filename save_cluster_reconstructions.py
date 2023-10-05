import ast
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
model_name = 'DMVAE'

fa_factor = [target_size / float(fa_size) for
             fa_size, target_size
             in zip((5, 121, 145, 121), (5, 182, 218, 182))]

cluster_differences = {
    'fBIRNFNCFA': 3,
    'fBIRNFNCsMRI': 0,
    'fBIRNICAFNC': 2,
    'fBIRNICAFA': 1,
    'fBIRNFAsMRI': 2,
    'fBIRNICAsMRI': 4
}

for dataset in cluster_differences.keys():
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

    fold = 0
    fold_path = Path(f'logs/{dataset}_{num_features}_{priv_size}_{shared_size}_{beta}_{str(model)}_1e-05_{str(encoder1)}_{str(encoder2)}{str(decoder1)}_{str(decoder2)}_{criterion_name}_6_42_300_new/fold_{fold}')
    print(fold_path)

    clusters = np.load(fold_path / 'clusters.npy')

    loaders, pipes = dataset_generator.build_pipes(fold)

    train_loaders = {"train": loaders['train'],
                    "valid": loaders['valid']}

    checkpoint = torch.load(fold_path / Path('best.pth'), map_location=device)

    print(f'Fold: {fold}, best epoch: {checkpoint["global_epoch_step"]}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

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
    tr_dist = torch.cat((tr_dist_tuple), dim=-1)
    va_dist = torch.cat((va_dist_tuple), dim=-1)
    te_dist = torch.cat((te_dist_tuple), dim=-1)
    total_dists = D.Normal(
        torch.cat((tr_dist[:, 0], va_dist[:, 0], te_dist[:, 0]), dim=0),
        torch.cat((tr_dist[:, 1], va_dist[:, 1], te_dist[:, 1]), dim=0),
    )

    total_mean = total_dists.mean.cpu().numpy()

    train_df = dataset_generator.dataframes[fold]['train']
    valid_df = dataset_generator.dataframes[fold]['valid']
    test_df = dataset_generator.dataframes[fold]['test']
    fold_df = pd.concat((train_df, valid_df, test_df), axis=0)

    assignment_df = pd.read_csv(f'assignments/{dataset.lower()}_assignments.csv', index_col=0)
    assignment_df.columns = assignment_df.columns.map(int)

    cluster_ix = cluster_differences[dataset]

    # Get assigned subjects for the cluster for fold 0
    cluster_subjects = ast.literal_eval(assignment_df.loc[cluster_ix, 0])
    print(fold_df.shape)
    sz_space = total_mean[fold_df['idc'].isin(cluster_subjects) & (fold_df['sz'] == 1)]
    cluster = sz_space.mean(0)

    cluster = torch.from_numpy(cluster).to(device).float().unsqueeze(0)
    rec_m1, rec_m2 = model.reconstruct_cluster(cluster)
    rec_m1, rec_m2 = rec_m1.cpu().numpy(), rec_m2.cpu().numpy()
    rec_m1, rec_m2 = reshape_data(dataset, rec_m1, rec_m2)
    if dataset == 'fBIRNFNCFA' or dataset == 'fBIRNICAFA':
        rec_m1 = ndimage.zoom(rec_m1, zoom=fa_factor)
    elif dataset == 'fBIRNFAsMRI':
        rec_m2 = ndimage.zoom(rec_m2, zoom=fa_factor)
    diff_m1 = rec_m1[0]
    diff_m2 = rec_m2[0]
    print(f'--- Dataset: {dataset} max abs ---')
    print(np.abs(diff_m1).max(), np.abs(diff_m2).max())
    print('----')
    save_recs('clustering/cluster', dataset, diff_m1, diff_m2, cutoff=0.025)
    print(f'--- Dataset: {dataset}, difference saved ---')
