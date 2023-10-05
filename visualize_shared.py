import gif
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
template = nb.load('mni152.nii.gz')

priv_size = 16
shared_size = 32
beta = 1.0
batch_size = 5
num_features = 8


datasets = ['fBIRNICAFA', 'fBIRNFNCsMRI', 'fBIRNFAsMRI', 'fBIRNFNCFA', 'fBIRNICAFNC', 'fBIRNICAsMRI']
model_name = 'DMVAE'
num_clusters = 6
fold = 0

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

    optimizer = criterion = epochs = callbacks = logdir = fold_mean = fold_var = scheduler = None

    init_train_df = dataset_generator.dataframes[0]['train']
    init_valid_df = dataset_generator.dataframes[0]['valid']
    init_test_df = dataset_generator.dataframes[0]['test']
    init_df = pd.concat((init_train_df, init_valid_df, init_test_df), axis=0)

    num_folds = 10
    fold0_latents = np.zeros((225, shared_size))
    corrs = np.zeros((shared_size, num_folds))
    similarity_folds = np.zeros((shared_size, num_folds))
    fold = 0
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
        model.eval()

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
        shared = np.concatenate((train_shared, valid_shared, test_shared), axis=0)
        
        priv_m1 = torch.cat((train_tuple[0], valid_tuple[0], test_tuple[0]), axis=0)
        priv_m2 = torch.cat((train_tuple[1], valid_tuple[1], test_tuple[1]), axis=0)
        
        # Load all the targets
        y_train = train_targets.cpu().numpy()
        y_valid = valid_targets.cpu().numpy()
        y_test = test_targets.cpu().numpy()
        y = np.concatenate((y_train, y_valid, y_test), axis=0)

        similarity_mean = torch.from_numpy(
            np.load(f'shared_analysis/{dataset.lower()}_similarity.npy')).to(device)
        corrs_mean = np.load(f'shared_analysis/{dataset.lower()}_correlations.npy')
        minmax_np = np.load(f'shared_analysis/{dataset.lower()}_minmax.npy')
        minmax = torch.from_numpy(minmax_np).to(device)
        num_latents = int(((similarity_mean > 0.70).cpu() & (np.abs(corrs_mean) > 0.3)).sum())
        print(f'Dataset: {dataset}, num latents: {num_latents}')
        latent_ix = ((similarity_mean > 0.70).cpu() & (np.abs(corrs_mean) > 0.3)).nonzero()
        priv_m1_avg = priv_m1.mean(0).to(device)
        priv_m2_avg = priv_m2.mean(0).to(device)
        if num_latents > 0:
            with torch.no_grad():
                int_m1, int_m2 = model.interpolate_shared(latent_ix.squeeze(), minmax, (priv_m1_avg, priv_m2_avg))
            int_m1 = int_m1.cpu().numpy()
            int_m2 = int_m2.cpu().numpy()
            for k in range(num_latents):
                int_m1_k, int_m2_k = reshape_data(dataset, int_m1[k], int_m2[k])
                plot_interpolation((dataset, latent_ix[k][0]), (int_m1_k, int_m2_k), (shared[:, latent_ix[k]], y, corrs_mean[latent_ix[k]]))
                print('saved fig')
        else:
            pass
