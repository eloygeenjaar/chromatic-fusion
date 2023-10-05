import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


random.seed(42)
np.random.seed(42)

datasets = ['fBIRNFNCFA', 'fBIRNFNCsMRI', 'fBIRNICAFNC', 'fBIRNICAFA', 'fBIRNFAsMRI', 'fBIRNICAsMRI']
dataset_names = ['FA-sFNC', 'sMRI-sFNC', 'ICA-sFNC', 'FA-ICA', 'sMRI-FA', 'sMRI-ICA']
corr_arr = np.zeros((32, len(datasets)))
for (d, dataset) in enumerate(datasets):
    similarity_mean = np.load(f'shared_analysis/{dataset.lower()}_similarity.npy')
    corrs_mean = np.load(f'shared_analysis/{dataset.lower()}_correlations.npy')

    corr_arr[similarity_mean >= 0.7, d] = corrs_mean[similarity_mean >= 0.7]

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
sns.heatmap(corr_arr.T, cmap='jet', square=True, yticklabels=dataset_names, ax=ax, fmt='g',
            vmin=-np.abs(corr_arr).max(), vmax=np.abs(corr_arr).max(), cbar_kws={"shrink": 0.4})
ax.set_xlabel('Latent dimensions')
plt.savefig('shared_appendix.png',bbox_inches='tight')
