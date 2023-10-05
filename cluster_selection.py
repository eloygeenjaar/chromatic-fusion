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
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from kmeans_jsd import KMeansJSD
from utils import *
from sklearn.metrics import silhouette_score


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

datasets = ['fBIRNICAFA', 'fBIRNFNCsMRI', 'fBIRNFAsMRI', 'fBIRNFNCFA', 'fBIRNICAFNC', 'fBIRNICAsMRI']
model_name = 'DMVAE'
cluster_perf = np.zeros((len(datasets), 10, 10))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for (d, dataset) in enumerate(datasets):
    # Load the dataframe with the paths and target variables
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

    # Performance for num clusters

    for fold in range(10):
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

            # Load all the targets
            y_train = train_targets.cpu().numpy()
            y_valid = valid_targets.cpu().numpy()
            train_targets = np.concatenate((y_train, y_valid), axis=0)
            y_test = test_targets.cpu().numpy()
            y_all = np.concatenate((y_train, y_valid, y_test), axis=0)

            for (i, num_clusters) in enumerate(range(2, 12)):
                kmeans = KMeans(n_clusters=num_clusters, n_init='auto', max_iter=1000, random_state=42)
                kmeans.fit(total_dists.mean)

                cluster_perf[d, i, fold] = kmeans.inertia_
                print(f'Dataset: {dataset}, clusters: {num_clusters}, fold: {fold}, score: {cluster_perf[d, i, fold]}')
                del kmeans
    dataset_perf = cluster_perf[d].mean(-1)
    print(dataset_perf)
    ax.plot(list(range(2, 12)), dataset_perf, alpha=0.8)
    ax.scatter([cluster_selection[dataset]], [dataset_perf[cluster_selection[dataset]-2]], marker='X', label='_nolegend_')
    ax.set_xlabel('The number of meta-chromatic patterns (MCPs)')
    ax.set_ylabel('Inertia')

ax.legend(datasets)
plt.savefig('cluster-selection.png')
