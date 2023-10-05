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
from scipy import ndimage
from utils import *


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

comp_names = [
            'CAU1', 'SUB/HYPOT', 'PUT', 'CAU2', 'THA',

            'STG', 'MTG1',

            'PoCG1', 'L PoCG', 'ParaCL1', 'R PoCG', 'SPL1',
            'ParaCL2', 'PreCG', 'SPL', 'PoCG2',

            'CalcarineG', 'MOG', 'MTG2', 'CUN', 'R MOG',
            'FUG', 'IOG', 'LingualG', 'MTG3',

            'IPL1', 'INS', 'SMFG', 'IFG1', 'R IFG', 'MiFG1',
            'IPL2', 'R IPL', 'SMA', 'SFG', 'MiFG2', 'HiPP1',
            'L IPL', 'MCC', 'IFG2', 'MiFG3', 'HiPP2',

            'Pr1', 'Pr2', 'ACC1', 'PCC1', 'ACC2', 'Pr3', 'PCC2',

            'CB1', 'CB2', 'CB3', 'CB4']

domain_sizes = [5, 2, 9, 9, 17, 7, 4]

temp_factor = [target_size / float(rec_size) for
               rec_size, target_size
               in zip((121, 145, 121), (207, 256, 215))]

mni_temp_affine = np.array([[   0.73746312,    0.        ,    0.        ,  -75.7625351 ],
                            [   0.        ,    0.73746312,    0.        , -110.7625351 ],
                            [   0.        ,    0.        ,    0.73746312,  -71.7625351 ],
                            [   0.        ,    0.        ,    0.        ,    1.        ]])

template = nb.load('mni152.nii.gz')

priv_size = 16
shared_size = 32
subject_size = 0
beta = 1.0
latent_dim = shared_size + priv_size
batch_size = 5
num_features = 8


model_name = 'DMVAE'
cluster_selection = {
    'fBIRNFNCFA': 4,
    'fBIRNFNCsMRI': 3,
    'fBIRNICAFNC': 4,
    'fBIRNICAFA': 6,
    'fBIRNFAsMRI': 4,
    'fBIRNICAsMRI': 6
    }


datasets = ['fBIRNICAFA', 'fBIRNFNCsMRI', 'fBIRNFAsMRI', 'fBIRNFNCFA', 'fBIRNICAFNC', 'fBIRNICAsMRI']
for dataset in datasets:

    num_clusters = cluster_selection[dataset]

    info_df = pd.read_csv(Path(f'./info_df_{dataset.lower()}.csv'))
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

    fold = 0
    fold_path = Path(f'logs/{dataset}_{num_features}_{priv_size}_{shared_size}_{beta}_{str(model)}_1e-05_{str(encoder1)}_{str(encoder2)}{str(decoder1)}_{str(decoder2)}_{criterion_name}_6_42_300_new/fold_{fold}')
    print(fold_path)

    clusters = np.load(fold_path / 'clusters.npy')
    assignments = np.load(fold_path / 'assignments.npy')

    loaders, pipes = dataset_generator.build_pipes(fold)

    train_loaders = {"train": loaders['train'],
                    "valid": loaders['valid']}

    checkpoint = torch.load(fold_path / Path('best.pth'), map_location=device)

    print(f'Fold: {fold}, best epoch: {checkpoint["global_epoch_step"]}')
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = criterion = epochs = callbacks = logdir = scheduler = None

    # Initialize runner, but only to call evaluate_loader
    runner = get_runner(model_name, model, train_loaders, optimizer,
            criterion, epochs, callbacks,
            scheduler, logdir, device)

    # These are the names of the subspaces
    names = ('p1', 'p2', 's', 's1', 's2', 'subj')

    # Run inference on the training, validation, and test set for a certain fold
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
    x_test = np.concatenate((test_tuple[0].cpu().numpy(), test_tuple[1].cpu().numpy(), test_tuple[2].cpu().numpy()), axis=1)
    tr_dist = torch.cat((tr_dist_tuple), dim=-1)
    va_dist = torch.cat((va_dist_tuple), dim=-1)
    te_dist = torch.cat((te_dist_tuple), dim=-1)
    total_dists = D.Normal(
        torch.cat((tr_dist[:, 0], va_dist[:, 0], te_dist[:, 0]), dim=0),
        torch.cat((tr_dist[:, 1], va_dist[:, 1], te_dist[:, 1]), dim=0),
    )

    total_mean = total_dists.mean

    # Load all the targets
    y_train = train_targets.cpu().numpy()
    y_valid = valid_targets.cpu().numpy()
    train_targets = np.concatenate((y_train, y_valid), axis=0)
    y_test = test_targets.cpu().numpy()
    y_all = np.concatenate((y_train, y_valid, y_test), axis=0)

    # Obtain the number of clusters that were found
    num_clusters = clusters.shape[0]

    # Calculate the percentage of patients in each cluster
    percentages = []
    for c in range(num_clusters):
        perc = (assignments[y_all==1] == c).sum() / (assignments == c).sum()
        percentages.append(str(np.round(perc, 2)))
        print(f'Cluster: {c}, percentage SZ: {perc}')

    # For the first fold, create a t-SNE plot
    if fold == 0 and model_name == 'DMVAE':            
        perplexities = [20, 30, 50]
        for perplex in perplexities:
            # Instantiate t-SNE for each complexity
            tsne = TSNE(perplexity=perplex, learning_rate='auto', init='pca')

            # Concatenate the averages of each subject and the average cluster for the KMeans algorithm
            embs_kmeans = np.concatenate((x_train, x_valid, x_test, clusters), axis=0)

            # Instantiate the red, green, and blue parts of the clusters (private-1, shared, private-2)
            # and how large their L2 distances are from the prior (0-mean) is for each color aspect of each cluster.
            red_clusters = np.linalg.norm(clusters[:, :priv_size], ord=2, axis=-1)
            green_clusters = np.linalg.norm(clusters[:, priv_size:(priv_size + shared_size)], ord=2, axis=-1)
            blue_clusters = np.linalg.norm(clusters[:, (priv_size + shared_size):], ord=2, axis=-1)

            # Normalize the colors of the clusters using the maximum of that cluster.
            red_clusters = (red_clusters / red_clusters.max())[:, np.newaxis]
            green_clusters = (green_clusters / green_clusters.max())[:, np.newaxis]
            blue_clusters = (blue_clusters / blue_clusters.max())[:, np.newaxis]
            cluster_colors = np.concatenate((red_clusters, green_clusters, blue_clusters), axis=1)

            # Create t-SNE embeddings of the representations and the KMeans cluster centers
            tsne_embeddings = tsne.fit_transform(embs_kmeans)

            # Separate out the training, validation, test set, and the cluster centers
            tsne_train = tsne_embeddings[:y_train.shape[0]]
            tsne_valid = tsne_embeddings[y_train.shape[0]:y_train.shape[0] + y_valid.shape[0]]
            tsne_test = tsne_embeddings[y_train.shape[0] + y_valid.shape[0]:y_train.shape[0] + y_valid.shape[0] + y_test.shape[0]]
            tsne_clusters = tsne_embeddings[y_train.shape[0] + y_valid.shape[0] + y_test.shape[0]:]

            # Instantiate an object to plot the t-SNE plot
            fig, ax = plt.subplots()
            fig.set_size_inches(12, 12)
            tsne_all = tsne_embeddings[:y_train.shape[0] + y_valid.shape[0] + y_test.shape[0]]

            # Calculate normalized distances to each cluster
            dist_matrix = np.linalg.norm(total_mean[:, None, :] - clusters[None, :, :], ord=2, axis=-1)
            dist_matrix = 1 / (dist_matrix ** 4)
            norm_dists = (dist_matrix / dist_matrix.sum(1, keepdims=True))

            # Create probabilities based on the distance between the points and the clusters
            point_colors = norm_dists @ cluster_colors

            #  Extract the full dataframes and specifically the column with sex stored
            percentages_sex = []
            df_all = np.concatenate(
                (dataset_generator.dataframes[fold]['train'].index.values,
                    dataset_generator.dataframes[fold]['valid'].index.values,
                    dataset_generator.dataframes[fold]['test'].index.values), axis=0)
            sex_all = np.concatenate(
                (dataset_generator.dataframes[fold]['train']['sex'].values,
                    dataset_generator.dataframes[fold]['valid']['sex'].values,
                    dataset_generator.dataframes[fold]['test']['sex'].values), axis=0)

            # Calculate the percentage of females in a cluster
            for c in range(num_clusters):
                perc_sex = (assignments[sex_all==1] == c).sum() / (assignments == c).sum()
                percentages_sex.append(str(np.round(perc_sex, 2)))

            # Plot four sets: (SZ & F, SZ & M, C & F, C & M)
            ax.scatter(tsne_all[(y_all == 1) & (sex_all == 1), 0],
                        tsne_all[(y_all == 1) & (sex_all == 1), 1],
                        alpha=0.9,
                        marker='P',
                        color=point_colors[(y_all == 1) & (sex_all == 1)],
                        edgecolors='k',
                        s=80)
            ax.scatter(tsne_all[(y_all == 0) & (sex_all == 1), 0],
                        tsne_all[(y_all == 0) & (sex_all == 1), 1],
                        alpha=0.9,
                        marker='X',
                        color=point_colors[(y_all == 0) & (sex_all == 1)],
                        edgecolors='k',
                        s=80)
            ax.scatter(tsne_all[(y_all == 1) & (sex_all == 0), 0],
                        tsne_all[(y_all == 1) & (sex_all == 0), 1],
                        alpha=0.9,
                        marker='o',
                        color=point_colors[(y_all == 1) & (sex_all == 0)],
                        edgecolors='k',
                        s=80)
            ax.scatter(tsne_all[(y_all == 0) & (sex_all == 0), 0],
                        tsne_all[(y_all == 0) & (sex_all == 0), 1],
                        alpha=0.9,
                        marker='D',
                        color=point_colors[(y_all == 0) & (sex_all == 0)],
                        edgecolors='k',
                        s=80)
            ax.scatter(tsne_clusters[:, 0],
                        tsne_clusters[:, 1],
                        color=cluster_colors,
                        marker='s',
                        edgecolors='k',
                        linewidth=2,
                        s=500)

            # For each of the clusters add text stating the number of the cluster
            for i in range(tsne_clusters.shape[0]):
                ax.text(tsne_clusters[i, 0], tsne_clusters[i, 1], s=str(i), color='k',
                            fontsize=16, horizontalalignment='center', multialignment='center',
                            fontweight='heavy',
                            verticalalignment='center')

            # Add the legend to the figure and save it
            plt.legend(['F-SZ', 'F-C', 'M-SZ', 'M-C'])
            plt.axis('off')
            plt.savefig(Path(f'results/{model_name.lower()}/{dataset.lower()}/clustering/sz_{perplex}.png'), dpi=200)
            print(f'--- Visualization saved to disk ---')
            plt.clf()

            # Create a nunmpy array of the sz and sex percentages
            percs_np = np.asarray([percentages, percentages_sex]).T

            # Add the cluster colors to the percentages
            percs_arr = np.concatenate((percs_np, cluster_colors), axis=1)

            # Create a dataframe and save it
            df = pd.DataFrame(percs_arr, columns=['sz', 'sex', 'red', 'green', 'blue'])
            df.to_csv(Path(f'results/{model_name.lower()}/{dataset.lower()}/clustering/cluster_info.csv'))

            # Specifically save the cluster colors as well
            cluster_arr = np.concatenate((np.arange(num_clusters)[:, np.newaxis], cluster_colors), axis=1)
            ix_sort = df['sz'].argsort().values
            cluster_arr = cluster_arr[ix_sort[::-1]]
            np.savetxt(f'results/{model_name.lower()}/{dataset.lower()}/clustering/cluster_colors.txt', cluster_arr, delimiter=',',
                        fmt='%1.3f')
