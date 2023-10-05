import gif
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from chromatic.architectures import MLP2DEncoder, MLP2DDecoder, DCGANEncoder, DCGANDecoder, DCGANEncoderICA, DCGANDecoderICA
from chromatic.models import DMVAE
from chromatic.datasets import *
from chromatic.runners import DMVAERunner
from pathlib import Path
from nilearn.image import resample_to_img, resample_img, new_img_like

comp_names = [
            'CAU1', 'SUB/HYPOT', 'PUT', 'CAU2', 'THA',

            'STG', 'MTG1',

            'PoCG1', 'L PoCG', 'ParaCL1', 'R PoCG', 'SPL1',
            'ParaCL2', 'PreCG', 'SPL', 'PoCG2',

            'CalcarineG', 'MOG', 'MTG2', 'CUN', 'R MOG',
            'FUG', 'IOG', 'LingualG', 'MTG3',

            'IPL1', 'INS', 'SMFG', 'IFG1', 'R IFG', 'MiFG1',
            'IPL2', 'R IPL', 'SMA', 'SFG', 'MiFG2', 'HiPP1'
            'L IPL', 'MCC', 'IFG2', 'MiFG3', 'HiPP2',

            'Pr1', 'Pr2', 'ACC1', 'PCC1', 'ACC2', 'Pr3', 'PCC2',

            'CB1', 'CB2', 'CB3', 'CB4']

domain_sizes = [5, 2, 9, 9, 17, 7, 4]


def get_dataset_generator(info_df, dataset, batch_size=5):
    if dataset == 'fBIRNFNCFA':
        dataset_generator = fBIRNFNCFA(info_df=info_df,
                                    target_names=['sz', 'age', 'sex'],
                                    main_target='sz', numpy_root=Path(f'/data/users1/egeenjaar/chromatic-fusion/Data/{dataset}'),
                                    preprocess=True, seed=42,
                                    num_folds=10,
                                    batch_size=batch_size)
    elif dataset == 'fBIRNFNCsMRI':
        dataset_generator = fBIRNFNCsMRI(info_df=info_df,
                                    target_names=['sz', 'age', 'sex'],
                                    main_target='sz', numpy_root=Path(f'/data/users1/egeenjaar/chromatic-fusion/Data/{dataset}'),
                                    preprocess=True, seed=42,
                                    num_folds=10,
                                    batch_size=batch_size)
    elif dataset == 'fBIRNICAsMRI':
        dataset_generator = fBIRNICAsMRI(info_df=info_df,
                                    target_names=['sz', 'age', 'sex'],
                                    main_target='sz', numpy_root=Path(f'/data/users1/egeenjaar/chromatic-fusion/Data/{dataset}'),
                                    preprocess=True, seed=42,
                                    num_folds=10,
                                    batch_size=batch_size)
    elif dataset == 'fBIRNFAsMRI':
        dataset_generator = fBIRNFAsMRI(info_df=info_df,
                                    target_names=['sz', 'age', 'sex'],
                                    main_target='sz', numpy_root=Path(f'/data/users1/egeenjaar/chromatic-fusion/Data/{dataset}'),
                                    preprocess=True, seed=42,
                                    num_folds=10,
                                    batch_size=batch_size)
    elif dataset == 'fBIRNICAFA':
        dataset_generator = fBIRNICAFA(info_df=info_df,
                                    target_names=['sz', 'age', 'sex'],
                                    main_target='sz', numpy_root=Path(f'/data/users1/egeenjaar/chromatic-fusion/Data/{dataset}'),
                                    preprocess=True, seed=42,
                                    num_folds=10,
                                    batch_size=batch_size)
    elif dataset == 'fBIRNICAFNC':
        dataset_generator = fBIRNICAFNC(info_df=info_df,
                                    target_names=['sz', 'age', 'sex'],
                                    main_target='sz', numpy_root=Path(f'/data/users1/egeenjaar/chromatic-fusion/Data/{dataset}'),
                                    preprocess=True, seed=42,
                                    num_folds=10,
                                    batch_size=batch_size)
    return dataset_generator

def get_architectures(dataset, input_shape, priv_size, shared_size, num_features):
    if dataset in ['fBIRNICAFNC']:
        encoder1 = DCGANEncoderICA(input_shape=input_shape,
                                priv_size=priv_size,
                                shared_size=shared_size,
                                num_features=num_features)
        decoder1 = DCGANDecoderICA(input_shape=input_shape,
                                private_dim=priv_size,
                                shared_dim=shared_size,
                                num_features=num_features)
    else:
        encoder1 = DCGANEncoder(input_shape=input_shape,
                                priv_size=priv_size,
                                shared_size=shared_size,
                                num_features=num_features)
        decoder1 = DCGANDecoder(input_shape=input_shape,
                                private_dim=priv_size,
                                shared_dim=shared_size,
                                num_features=num_features)
    if dataset in ['fBIRNFNCsMRI', 'fBIRNFNCFA', 'fBIRNICAFNC']:
        encoder2 = MLP2DEncoder(input_shape=input_shape,
                                priv_size=priv_size,
                                shared_size=shared_size,
                                num_features=num_features)
        decoder2 = MLP2DDecoder(input_shape=input_shape,
                                private_dim=priv_size,
                                shared_dim=shared_size,
                                num_features=num_features)
    elif dataset in ['fBIRNICAsMRI', 'fBIRNICAFA']:
        encoder2 = DCGANEncoderICA(input_shape=input_shape,
                                priv_size=priv_size,
                                shared_size=shared_size,
                                num_features=num_features)
        decoder2 = DCGANDecoderICA(input_shape=input_shape,
                                private_dim=priv_size,
                                shared_dim=shared_size,
                                num_features=num_features)
    elif dataset in ['fBIRNFAsMRI']:
        encoder2 = DCGANEncoder(input_shape=input_shape,
                                priv_size=priv_size,
                                shared_size=shared_size,
                                num_features=num_features)
        decoder2 = DCGANDecoder(input_shape=input_shape,
                                private_dim=priv_size,
                                shared_dim=shared_size,
                                num_features=num_features)
    return encoder1, encoder2, decoder1, decoder2

def get_model(model_name, input_shape, encoder1, encoder2, decoder1, decoder2, dist, dataset, device):
    model = DMVAE(input_shape=input_shape,
                        encoder1=encoder1,
                        decoder1=decoder1,
                        encoder2=encoder2,
                        decoder2=decoder2,
                        post_dist=dist,
                        likelihood_dist=dist,
                        dataset=dataset).to(device)
    criterion_name = 'ELBO'
    return model, criterion_name

def get_runner(model_name, model, train_loaders, optimizer,
               criterion, epochs, callbacks,
               scheduler, logdir, device):
    runner = DMVAERunner(
                model=model,
                loaders=train_loaders,
                optimizer=optimizer,
                criterion=criterion,
                epochs=epochs,
                callbacks=callbacks,
                scheduler=scheduler,
                logdir=logdir,
                device=device)
    return runner

def embed_data(dataset, loaders, data_type):
    if dataset == 'fBIRNFAsMRI':
        m1_arr = torch.zeros((len(loaders[data_type].dataset), 1, 121, 145, 121))
        m2_arr = torch.zeros((len(loaders[data_type].dataset), 1, 121, 145, 121))
    y_arr = torch.zeros((len(loaders[data_type].dataset), ), dtype=torch.float)
    ix_start = 0
    for (i, batch) in enumerate(loaders[data_type]):
        m1 = batch[0]['m1'].float()
        m2 = batch[0]['m2'].float()
        ix_end = ix_start + m1.size(0)
        m1_arr[ix_start:ix_end] = m1
        m2_arr[ix_start:ix_end] = m2
        y_arr[ix_start:ix_end] = batch[0]['targets']
        ix_start = ix_end
    m1_arr = np.reshape(m1_arr.cpu().numpy(), (m1_arr.shape[0], -1))
    m2_arr = np.reshape(m2_arr.cpu().numpy(), (m2_arr.shape[0], -1))
    y = y_arr.cpu().numpy()
    x = np.concatenate((m1_arr, m2_arr), axis=1)
    return (x, y)

def reshape_data(dataset, m1, m2):
    if dataset == 'fBIRNFNCFA':
        m1 = np.reshape(m1, (m1.shape[0], 121, 145, 121))
        m2_ls = []
        for i in range(m2.shape[0]):
            new_m2 = np.zeros((53, 53))
            new_m2[np.triu_indices(53, 1)] = m2[i]
            new_m2 = np.triu(new_m2) + np.triu(new_m2).T
            m2_ls.append(new_m2)
        m2 = np.stack(m2_ls, axis=0)
    elif dataset == 'fBIRNFNCsMRI':
        m1 = np.reshape(m1, (m1.shape[0], 121, 145, 121))
        m2_ls = []
        for i in range(m2.shape[0]):
            new_m2 = np.zeros((53, 53))
            new_m2[np.triu_indices(53, 1)] = m2[i]
            new_m2 = np.triu(new_m2) + np.triu(new_m2).T
            m2_ls.append(new_m2)
        m2 = np.stack(m2_ls, axis=0)
    elif dataset == 'fBIRNICAsMRI':
        m1 = np.reshape(m1, (m2.shape[0], 121, 145, 121))
        m2 = np.reshape(m2, (m2.shape[0], 8, 53, 63, 52)).sum(1)
    elif dataset == 'fBIRNFAsMRI':
        m1 = np.reshape(m1, (m1.shape[0], 121, 145, 121))
        m2 = np.reshape(m2, (m2.shape[0], 121, 145, 121))
    elif dataset == 'fBIRNICAFA':
        m1 = np.reshape(m1, (m1.shape[0], 121, 145, 121))
        m2 = np.reshape(m2, (m2.shape[0], 8, 53, 63, 52)).sum(1)
    elif dataset == 'fBIRNICAFNC':
        m1 = np.reshape(m1, (m1.shape[0], 8, 53, 63, 52)).sum(1)
        m2_ls = []
        for i in range(m2.shape[0]):
            new_m2 = np.zeros((53, 53))
            new_m2[np.triu_indices(53, 1)] = m2[i]
            new_m2 = np.triu(new_m2) + np.triu(new_m2).T
            m2_ls.append(new_m2)
        m2 = np.stack(m2_ls, axis=0)
    return m1, m2

def plot_interpolation(save_info, int_modalities, shared):
    int_m1, int_m2 = int_modalities
    plots = []
    noise_c = np.random.rand((shared[1]==0).sum())
    noise_sz = np.random.rand((shared[1]==1).sum())
    vmax_m1 = np.max(np.array([np.abs(np.min(int_m1)), np.abs(np.max(int_m1))]))
    vmin_m1 = -vmax_m1
    vmax_m2 = np.max(np.array([np.abs(np.min(int_m2)), np.abs(np.max(int_m2))]))
    vmin_m2 = -vmax_m2
    int_m1 = form_3x3(int_m1)
    if int_m2.shape[-1] != 53:
        int_m2 = form_3x3(int_m2)
    save_interpolation(save_info, (int_m1, int_m2), shared, ((vmin_m1, vmax_m1), (vmin_m2, vmax_m2)),
                       (noise_c, noise_sz))

def form_3x3(modality):
    _, size_x, size_y, size_z = modality.shape
    print(size_z)
    if size_z == 52:
        base = 10
    else:
        base = 20
    step = (size_z - 2 * base) // 9
    new_mod = np.zeros((modality.shape[0], 3 * modality.shape[1], 3 * modality.shape[2]))
    for i in range(3):
        for j in range(3):
            new_mod[:, i * size_x:(i + 1) * size_x, j * size_y:(j + 1) * size_y] = (
                modality[..., base + (i * 3 + j) * step]
            )
    return np.transpose(new_mod, axes=(0, 2, 1))

def save_interpolation(save_info, modalities, sz, ranges, noise):
    dataset, latent_ix = save_info
    fig, ax = plt.subplots(2, 10, figsize=(20, 10))
    noise_c, noise_sz = noise
    for j in range(10):
        ax_m1 = ax[0, j]
        ax_m2 = ax[1, j]
        int_m1, int_m2 = modalities
        ((vmin_m1, vmax_m1), (vmin_m2, vmax_m2)) = ranges
        ax_m1.imshow(int_m1[j], vmin=vmin_m1, vmax=vmax_m1, cmap='jet')
        ax_m1.axis('off')
        if int_m2.shape[-1] != 53:
            int_m2[j] = np.ma.masked_where(int_m2[j] <= 1E-3, int_m2[j])
            ax_m2.imshow(int_m2[j], vmin=vmin_m2, vmax=vmax_m2, cmap='jet')
            ax_m2.axis('off')
        else:
            ax_m2.imshow(int_m2[j], vmin=vmin_m2, vmax=vmax_m2, cmap='jet', extent=[0, 53, 0, 53])
            cur_size = 0
            for domain_size in domain_sizes[::-1]:
                cur_size += domain_size
                ax_m2.plot([0, 53], [cur_size, cur_size], c='k', linewidth=0.5 ,alpha=0.5)
                ax_m2.plot([53-cur_size, 53-cur_size], [53, 0], c='k', linewidth=0.5 ,alpha=0.5)
            ax_m2.set_yticks(np.arange(len(comp_names)))
            ax_m2.set_yticklabels(comp_names)
            ax_m2.tick_params(axis='both', which='major', labelsize=6)
            ax_m2.axis('off')
            ax_m2.set_aspect('equal')
    shared, y, corr_mean = sz
    fig.suptitle(f'Correlation: {np.round(corr_mean, 2)}')
    plt.tight_layout()
    plt.savefig(f'shared_analysis/results/{dataset.lower()}/latent_{latent_ix}.png', dpi=400)
    plt.clf()
    plt.close(fig)


@gif.frame
def plot_frame(modalities, sz, ranges, noise):
    fig = plt.figure()
    gs = fig.add_gridspec(3, 10)
    noise_c, noise_sz = noise
    for j in range(10):
        ax_m1 = fig.add_subplot(gs[0, j])
        ax_m2 = fig.add_subplot(gs[1, j])
        ax3 = fig.add_subplot(gs[2, :])
        int_m1, int_m2 = modalities
        ((vmin_m1, vmax_m1), (vmin_m2, vmax_m2)) = ranges
        ax_m1.imshow(int_m1[j], vmin=vmin_m1, vmax=vmax_m1, cmap='jet')
        ax_m1.axis('off')
        if int_m2.shape[-1] != 53:
            ax_m2.imshow(int_m2[j], vmin=vmin_m2, vmax=vmax_m2, cmap='jet')
            ax_m2.axis('off')
        else:
            ax_m2.imshow(int_m2[j], vmin=vmin_m2, vmax=vmax_m2, cmap='jet', extent=[0, 53, 0, 53])
            cur_size = 0
            for domain_size in domain_sizes[::-1]:
                cur_size += domain_size
                ax_m2.plot([0, 53], [cur_size, cur_size], c='k', linewidth=0.1 ,alpha=0.5)
                ax_m2.plot([53-cur_size, 53-cur_size], [53, 0], c='k', linewidth=0.1 ,alpha=0.5)
            ax_m2.set_yticks(np.arange(len(comp_names)))
            ax_m2.set_yticklabels(comp_names)
            ax_m2.tick_params(axis='both', which='major', labelsize=6)
            ax_m2.axis('off')
            ax_m2.set_aspect('equal')
    shared, y, corr_mean = sz
    ax3.scatter(shared[(y==0)], noise_c, color='b', alpha=0.5, s=5)
    ax3.scatter(shared[(y==1)], noise_sz, color='r', alpha=0.5, s=5)
    ax3.axis('off')
    fig.suptitle(f'Correlation: {np.round(corr_mean[0], 2)}')

def load_fold_statistics(dataset, fold):
    if dataset == 'fBIRNFNCFA':
        m1_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_fa_mean.npy')
        m1_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_fa_std.npy')
        m2_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_fnc_mean.npy')
        m2_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_fnc_std.npy')
    elif dataset == 'fBIRNFNCsMRI':
        m1_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_smri_mean.npy')
        m1_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_smri_std.npy')
        m2_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_fnc_mean.npy')
        m2_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_fnc_std.npy')
    elif dataset == 'fBIRNICAsMRI':
        m1_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_smri_mean.npy')
        m1_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_smri_std.npy')
        m2_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_ica_mean.npy')
        m2_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_ica_std.npy')
    elif dataset == 'fBIRNFAsMRI':
        m1_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_smri_mean.npy')
        m1_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_smri_std.npy')
        m2_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_fa_mean.npy')
        m2_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_fa_std.npy')
    elif dataset == 'fBIRNICAFA':
        m1_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_fa_mean.npy')
        m1_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_fa_std.npy')
        m2_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_ica_mean.npy')
        m2_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_ica_std.npy')
    elif dataset == 'fBIRNICAFNC':
        m1_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_ica_mean.npy')
        m1_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_ica_std.npy')
        m2_fold_mean = np.load(f'./fold_means/{dataset}.lower()_{fold}_fnc_mean.npy')
        m2_fold_std = np.load(f'./fold_means/{dataset}.lower()_{fold}_fnc_std.npy')
    return (m1_fold_mean, m2_fold_mean), (m1_fold_std, m2_fold_std)

def resample_image(img):
    mni_template = nb.load('mni152.nii.gz')
    img = resample_img(img, target_affine=mni_template.affine)
    img = resample_to_img(img, mni_template)
    return img

def save_recs(name, dataset, rec_m1, rec_m2, cutoff=0.10):
    if dataset == 'fBIRNFNCFA':
        m1_template = nb.load('fa.nii.gz')
        m1_img = nb.Nifti1Image(rec_m1, affine=m1_template.affine)
        m1_img = resample_image(m1_img)
        m1_data = m1_img.get_fdata()
        m1_data = (np.abs(m1_data) > cutoff) * m1_data
        print(f'M1: {m1_data.min(), m1_data.max()}')
        print(f'M2: {rec_m2.min(), rec_m2.max()}')
        m1_img = new_img_like(m1_img, m1_data, copy_header=True)
        nb.save(m1_img, f'./results/dmvae/{dataset.lower()}/{name}_m1.nii')
        np.save(f'./results/dmvae/{dataset.lower()}/{name}_m2.npy', rec_m2)
    elif dataset == 'fBIRNFNCsMRI':
        m1_template = nb.load('smri.nii')
        m1_img = nb.Nifti1Image(rec_m1, affine=m1_template.affine)
        m1_img = resample_image(m1_img)
        m1_data = m1_img.get_fdata()
        m1_data = (np.abs(m1_data) > cutoff) * m1_data
        print(f'M1: {m1_data.min(), m1_data.max()}')
        print(f'M2: {rec_m2.min(), rec_m2.max()}')
        m1_img = new_img_like(m1_img, m1_data, copy_header=True)
        nb.save(m1_img, f'./results/dmvae/{dataset.lower()}/{name}_m1.nii')
        np.save(f'./results/dmvae/{dataset.lower()}/{name}_m2.npy', rec_m2)
    elif dataset == 'fBIRNICAsMRI':
        m1_template = nb.load('smri.nii')
        m1_img = nb.Nifti1Image(rec_m1, affine=m1_template.affine)
        m1_img = resample_image(m1_img)
        m1_data = m1_img.get_fdata()
        m1_data = (np.abs(m1_data) > cutoff) * m1_data
        m1_img = new_img_like(m1_img, m1_data, copy_header=True)
        nb.save(m1_img, f'./results/dmvae/{dataset.lower()}/{name}_m1.nii')
        m2_template = nb.load('ica.nii')
        m2_img = nb.Nifti1Image(rec_m2, affine=m2_template.affine)
        m2_img = resample_image(m2_img)
        m2_data = m2_img.get_fdata()
        m2_data = (np.abs(m2_data) > cutoff) * m2_data
        print(f'M1: {m1_data.min(), m1_data.max()}')
        print(f'M2: {m2_data.min(), m2_data.max()}')
        m2_img = new_img_like(m2_img, m2_data, copy_header=True)
        nb.save(m2_img, f'./results/dmvae/{dataset.lower()}/{name}_m2.nii')
    elif dataset == 'fBIRNFAsMRI':
        m1_template = nb.load('smri.nii')
        m1_img = nb.Nifti1Image(rec_m1, affine=m1_template.affine)
        m1_img = resample_image(m1_img)
        m1_data = m1_img.get_fdata()
        m1_data = (np.abs(m1_data) > cutoff) * m1_data
        m1_img = new_img_like(m1_img, m1_data, copy_header=True)
        nb.save(m1_img, f'./results/dmvae/{dataset.lower()}/{name}_m1.nii')
        m2_template = nb.load('fa.nii.gz')
        m2_img = nb.Nifti1Image(rec_m2, affine=m2_template.affine)
        m2_img = resample_image(m2_img)
        m2_data = m2_img.get_fdata()
        m2_data = (np.abs(m2_data) > cutoff) * m2_data
        print(f'M1: {m1_data.min(), m1_data.max()}')
        print(f'M2: {m2_data.min(), m2_data.max()}')
        m2_img = new_img_like(m2_img, m2_data, copy_header=True)
        nb.save(m2_img, f'./results/dmvae/{dataset.lower()}/{name}_m2.nii')
    elif dataset == 'fBIRNICAFA':
        m1_template = nb.load('fa.nii.gz')
        m1_img = nb.Nifti1Image(rec_m1, affine=m1_template.affine)
        m1_img = resample_image(m1_img)
        m1_data = m1_img.get_fdata()
        m1_data = (np.abs(m1_data) > cutoff) * m1_data
        m1_img = new_img_like(m1_img, m1_data, copy_header=True)
        nb.save(m1_img, f'./results/dmvae/{dataset.lower()}/{name}_m1.nii')
        m2_template = nb.load('ica.nii')
        m2_img = nb.Nifti1Image(rec_m2, affine=m2_template.affine)
        m2_img = resample_image(m2_img)
        m2_data = m2_img.get_fdata()
        m2_data = (np.abs(m2_data) > cutoff) * m2_data
        print(f'M1: {m1_data.min(), m1_data.max()}')
        print(f'M2: {m2_data.min(), m2_data.max()}')
        m2_img = new_img_like(m2_img, m2_data, copy_header=True)
        nb.save(m2_img, f'./results/dmvae/{dataset.lower()}/{name}_m2.nii')
    elif dataset == 'fBIRNICAFNC':
        m1_template = nb.load('ica.nii')
        m1_img = nb.Nifti1Image(rec_m1, affine=m1_template.affine)
        m1_img = resample_image(m1_img)
        m1_data = m1_img.get_fdata()
        m1_data = (np.abs(m1_data) > cutoff) * m1_data
        m1_img = new_img_like(m1_img, m1_data, copy_header=True)
        nb.save(m1_img, f'./results/dmvae/{dataset.lower()}/{name}_m1.nii')
        print(f'M1: {m1_data.min(), m1_data.max()}')
        print(f'M2: {rec_m2.min(), rec_m2.max()}')
        np.save(f'./results/dmvae/{dataset.lower()}/{name}_m2.npy', rec_m2)
