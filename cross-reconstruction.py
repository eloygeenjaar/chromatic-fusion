import torch
import pandas as pd
import numpy as np
import random
from torch import nn
from torch import distributions as D
from chromatic.architectures import *
from chromatic.models import DMVAE
from chromatic.datasets import *
from chromatic.runners import DMVAERunner
from torch import distributions as D
from pathlib import Path
from utils import *

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

priv_size = 16
shared_size = 32
subject_size = 0
beta = 1.0
latent_dim = shared_size + priv_size
batch_size = 5
num_features = 8
model_name = 'DMVAE'
datasets = ['fBIRNICAFA', 'fBIRNFNCsMRI', 'fBIRNFAsMRI', 'fBIRNFNCFA', 'fBIRNICAFNC', 'fBIRNICAsMRI']
mse_loss = nn.MSELoss(reduction='none')
for dataset in datasets:

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

    results = np.empty((10, 3, 2))
    for fold in range(10):
        fold_path = Path(f'logs/{dataset}_{num_features}_{priv_size}_{shared_size}_{beta}_{str(model)}_1e-05_{str(encoder1)}_{str(encoder2)}{str(decoder1)}_{str(decoder2)}_{criterion_name}_6_42_300_new/fold_{fold}')
        if fold_path.is_dir():
            
            loaders, pipes = dataset_generator.build_pipes(fold)
            
            checkpoint = torch.load(fold_path / Path('best.pth'), map_location=device)
            
            print(f'Fold: {fold}, best epoch: {checkpoint["global_epoch_step"]}')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            mse = nn.MSELoss(reduction='none')

            diff_m1_z = 0.
            diff_m1_s = 0.
            diff_m1_c = 0.
            diff_m2_z = 0.
            diff_m2_s = 0.
            diff_m2_c = 0.
            n = 0
            for (i, (batch)) in enumerate(loaders['test']):
                m1 = batch[0]['m1'].float()
                m2 = batch[0]['m2'].float()
                y = batch[0]['targets']
                
                with torch.no_grad():
                    out = model.cross_reconstruction(m1, m2)
                    rec_m1_z, rec_m2_z = out['rec_z']
                    rec_m1_s, rec_m2_s = out['rec_s']
                    rec_m1_c, rec_m2_c = out['rec_c']
                    m1, m2 = m1.view(m1.size(0), -1), m2.view(m2.size(0), -1)
                    mask_m1, mask_m2 = out['masks']
                    if mask_m1 is not None:
                        diff_m1_z += mse(rec_m1_z[:, mask_m1], m1[:, mask_m1]).mean(-1).sum(0)
                        diff_m1_s += mse(rec_m1_s[:, mask_m1], m1[:, mask_m1]).mean(-1).sum(0)
                        diff_m1_c += mse(rec_m1_c[:, mask_m1], m1[:, mask_m1]).mean(-1).sum(0)
                    else:
                        diff_m1_z += mse(rec_m1_z, m1).mean(-1).sum(0)
                        diff_m1_s += mse(rec_m1_s, m1).mean(-1).sum(0)
                        diff_m1_c += mse(rec_m1_c, m1).mean(-1).sum(0)
                    if mask_m2 is not None:
                        diff_m2_z += mse(rec_m2_z[:, mask_m2], m2[:, mask_m2]).mean(-1).sum(0)
                        diff_m2_s += mse(rec_m2_s[:, mask_m2], m2[:, mask_m2]).mean(-1).sum(0)
                        diff_m2_c += mse(rec_m2_c[:, mask_m2], m2[:, mask_m2]).mean(-1).sum(0)
                    else:
                        diff_m2_z += mse(rec_m2_z, m2).mean(-1).sum(0)
                        diff_m2_s += mse(rec_m2_s, m2).mean(-1).sum(0)
                        diff_m2_c += mse(rec_m2_c, m2).mean(-1).sum(0)
                    n += m1.size(0)
            diff_m1_z /= n
            diff_m1_s /= n
            diff_m1_c /= n
            diff_m2_z /= n
            diff_m2_s /= n
            diff_m2_c /= n
        
            results[fold, 0, 0] = diff_m1_z
            results[fold, 1, 0] = diff_m1_s
            results[fold, 2, 0] = diff_m1_c
            
            results[fold, 0, 1] = diff_m2_z
            results[fold, 1, 1] = diff_m2_s
            results[fold, 2, 1] = diff_m2_c

            print(f'Z loss: {diff_m1_z}, {diff_m2_z}')
            print(f'S loss: {diff_m1_s}, {diff_m2_s}')
            print(f'C loss: {diff_m1_c}, {diff_m2_c}')
        
    np.set_printoptions(precision=4, suppress=True)
    print(f'--- Dataset: {dataset} ---')
    print(f'--- Mean ---')
    print(np.mean(results, axis=0))
    print(f'--- SD ---')
    print(np.std(results, axis=0))
