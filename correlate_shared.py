import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random
import nibabel as nb
from torch import distributions as D
from pathlib import Path
from scipy.stats import pearsonr
from scipy import optimize
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


datasets = ['fBIRNFNCsMRI', 'fBIRNFAsMRI', 'fBIRNFNCFA', 'fBIRNICAFA', 'fBIRNICAFNC', 'fBIRNICAsMRI']
model_name = 'DMVAE'

for dataset in datasets:
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

    init_train_df = dataset_generator.dataframes[0]['train']
    init_valid_df = dataset_generator.dataframes[0]['valid']
    init_test_df = dataset_generator.dataframes[0]['test']
    init_df = pd.concat((init_train_df, init_valid_df, init_test_df), axis=0)

    num_folds = 10
    fold0_latents = np.zeros((225, shared_size))
    corrs = np.zeros((shared_size, num_folds))
    similarity_folds = np.zeros((shared_size, num_folds))
    for fold in range(num_folds):
        fold_path = Path(f'logs/{dataset}_{num_features}_{priv_size}_{shared_size}_{beta}_{str(model)}_1e-05_{str(encoder1)}_{str(encoder2)}{str(decoder1)}_{str(decoder2)}_{criterion_name}_6_42_300_new/fold_{fold}')
        fold_shared = pd.DataFrame(np.zeros((init_df.shape[0], shared_size)), index=init_df['idc'])
        fold_y = pd.Series(np.zeros((init_df.shape[0],)), index=init_df['idc'])
        if fold_path.is_dir():
            train_df = dataset_generator.dataframes[fold]['train']
            valid_df = dataset_generator.dataframes[fold]['valid']
            test_df = dataset_generator.dataframes[fold]['test']
            fold_df = pd.concat((train_df, valid_df, test_df), axis=0)
            
            # Create loaders
            loaders, pipes = dataset_generator.build_pipes(fold)

            train_loaders = {"train": loaders['train'],
                            "valid": loaders['valid']}
            
            # Load best checkpoint (based on validation performance)
            checkpoint = torch.load(fold_path / Path('best.pth'), map_location=device)
            
            print(f'Fold: {fold}, best epoch: {checkpoint["global_epoch_step"]}')
            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer = criterion = epochs = callbacks = logdir = fold_mean = fold_var = scheduler = None

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
            train_shared = train_tuple[2].cpu().numpy()
            valid_shared = valid_tuple[2].cpu().numpy()
            test_shared = test_tuple[2].cpu().numpy()

            # Load all the targets
            y_train = train_targets.cpu().numpy()
            y_valid = valid_targets.cpu().numpy()
            y_test = test_targets.cpu().numpy()
            
            shared = np.concatenate((train_shared, valid_shared, test_shared), axis=0)
            y = np.concatenate((y_train, y_valid, y_test), axis=0)
            
            fold_shared.loc[fold_df['idc'], :] = shared
            fold_y.loc[fold_df['idc']] = y
            
            shared = fold_shared.values.copy()
            y = fold_y.values.copy()

            if fold == 0:
                fold0_latents = shared.copy()
                minmax = np.stack((np.min(shared, axis=0), np.max(shared, axis=0)), axis=1)
                print(minmax.shape)
            else:
                corr_matrix = np.zeros((shared_size, shared_size))
                for i in range(shared_size):
                    for j in range(shared_size):
                        r, _ = pearsonr(fold0_latents[:, i], shared[:, j])
                        corr_matrix[i, j] = r
                row_ix, col_ix = optimize.linear_sum_assignment(-np.abs(corr_matrix))

                # Make sure the sign is the same as fold 0 so correlations 
                # with schizophrenia have the same sign
                shared = np.sign(corr_matrix[row_ix, col_ix]) * shared[:, col_ix].copy()

                similarity = np.abs(corr_matrix)[row_ix, col_ix]
                similarity_folds[:, fold] = similarity

            for i in range(shared_size):
                r, _ = pearsonr(shared[:, i], y)
                corrs[i, fold] = r

    print(f'Dataset: {dataset} ----')
    similarity_mean = np.mean(similarity_folds, axis=1)
    print(similarity_mean)
    corrs_mean = np.mean(corrs, axis=1)
    print(corrs_mean)

    np.save(f'shared_analysis/{dataset.lower()}_similarity.npy', similarity_mean)
    np.save(f'shared_analysis/{dataset.lower()}_correlations.npy', corrs_mean)
    np.save(f'shared_analysis/{dataset.lower()}_minmax.npy', minmax)
