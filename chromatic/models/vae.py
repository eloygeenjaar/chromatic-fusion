import torch  # noqa
import itertools
import numpy as np

from .basemodel import BaseModel
from chromatic.architectures import BaseArchitecture
from torch import nn  # noqa
from torch.nn import functional as F
from torch.nn.utils import weight_norm  # noqa

EPS = 1E-10
MAX_LOG_VAR = 2.18 # ~ 3SD (log(9))

class DMVAE(BaseModel):
    def __init__(self,
                 input_shape,
                 encoder1: BaseArchitecture,
                 encoder2: BaseArchitecture,
                 decoder1: BaseArchitecture,
                 decoder2: BaseArchitecture,
                 post_dist: torch.distributions.Distribution,
                 likelihood_dist: torch.distributions.Distribution,
                 dataset: str):
        super().__init__()

        self._input_shape = input_shape
        self._encoder1 = encoder1
        self._encoder2 = encoder2
        self._decoder1 = decoder1
        self._decoder2 = decoder2
        self._post_dist = post_dist
        self._likelihood_dist = likelihood_dist
        self._dataset = dataset
        
        self._mask1 = None
        self._mask2 = None
        if dataset == 'fBIRNFNCsMRI':
            self._mask1 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_smri_mask.npy'))
        elif dataset == 'fBIRNFNCFA':
            self._mask1 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_fa_mask.npy'))
        elif dataset == 'fBIRNICAsMRI':
            self._mask1 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_smri_mask.npy'))
            self._mask2 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_ica_mask.npy')).bool()
        elif dataset == 'fBIRNFAsMRI':
            self._mask1 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_smri_mask.npy'))
            self._mask2 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_fa_mask.npy'))
        elif dataset == 'fBIRNICAFA':
            self._mask1 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_fa_mask.npy'))
            self._mask2 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_ica_mask.npy')).bool()
        elif dataset == 'fBIRNICAFNC':
            self._mask1 = torch.from_numpy(np.load('/data/users1/egeenjaar/atlases/fbirn_ica_mask.npy')).bool()

        if self._mask1 is not None:
            self._mask1 = torch.reshape(self._mask1, (-1, ))
        if self._mask2 is not None:
            self._mask2 = torch.reshape(self._mask2, (-1, ))

        self._shared_size = self._encoder1.shared_size
        self._encoder1_size = self._encoder1.private_size + self._encoder1.shared_size
        self._encoder2_size = self._encoder2.private_size + self._encoder2.shared_size
        self._subj_size = self._encoder1.shared_size

        self._bn1 = nn.GroupNorm(8, self._encoder1_size)
        self._bn2 = nn.GroupNorm(8, self._encoder2_size)
        self._priv_m1_mean = nn.Linear(self._encoder1.private_size,
                                       self._encoder1.private_size, bias=False)
        self._priv_m1_lv = nn.Linear(self._encoder1.private_size,
                                       self._encoder1.private_size, bias=False)
        self._s_m1_mean = nn.Linear(self._encoder1.shared_size,
                                    self._encoder1.shared_size, bias=False)
        self._s_m1_lv = nn.Linear(self._encoder1.shared_size,
                                  self._encoder1.shared_size, bias=False)

        self._priv_m2_mean = nn.Linear(self._encoder2.private_size,
                                       self._encoder2.private_size, bias=False)
        self._priv_m2_lv = nn.Linear(self._encoder2.private_size,
                                     self._encoder2.private_size, bias=False)
        self._s_m2_mean = nn.Linear(self._encoder2.shared_size,
                                    self._encoder2.shared_size, bias=False)
        self._s_m2_lv = nn.Linear(self._encoder2.shared_size,
                                  self._encoder2.shared_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        self._act = nn.ELU()

        self.apply(self._init_weights)

    @property
    def latent_dim(self):
        return self._encoder2.private_size + self._shared_size + self._encoder1.private_size
    
    # Based on: https://github.com/mhw32/multimodal-vae-public/blob/master/multimnist/model.py
    def apply_poe(self, shared_mean_m1, shared_lv_m1, shared_mean_m2, shared_lv_m2):
        lv = torch.stack((shared_lv_m1, shared_lv_m2), dim=1)
        mu = torch.stack((shared_mean_m1, shared_mean_m2), dim=1)
        # Clamp the variance to avoid the shared std from blowing up
        var = torch.exp(lv)
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=1) / torch.sum(T, dim=1)
        pd_std = torch.sqrt(1 / torch.sum(T, dim=1))
        return pd_mu, pd_std

    def split_dims(self, latent_tensor):
        private_size = latent_tensor.size(1) - self._shared_size
        shared, private  = torch.split(
            latent_tensor, [self._shared_size, private_size], dim=1)
        return shared, private

    @property
    def private_size(self):
        return self._encoder1.private_size
    
    @property
    def shared_size(self):
        return self._shared_size

    @property
    def latent_dim(self):
        return self._subj_size


    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_keys(self):
        return ['dists',
                'recs',
                'embedding']
        
    def forward(self, m1, m2):
        batch_size = m1.size(0)

        m1 = self.dropout(self._act(self._bn1(self._encoder1(m1))))
        m2 = self.dropout(self._act(self._bn2(self._encoder2(m2))))
        
        shared_m1, priv_m1 = self.split_dims(m1)
        shared_m2, priv_m2 = self.split_dims(m2)
        
        priv_mean_m1, priv_lv_m1 = self._priv_m1_mean(priv_m1), self._priv_m1_lv(priv_m1).clamp(EPS, MAX_LOG_VAR)
        priv_mean_m2, priv_lv_m2 = self._priv_m2_mean(priv_m2), self._priv_m2_lv(priv_m2).clamp(EPS, MAX_LOG_VAR)
        shared_mean_m1, shared_lv_m1 = self._s_m1_mean(shared_m1), self._s_m1_lv(shared_m1).clamp(EPS, MAX_LOG_VAR)
        shared_mean_m2, shared_lv_m2 = self._s_m2_mean(shared_m2), self._s_m2_lv(shared_m2).clamp(EPS, MAX_LOG_VAR)
        
        shared_mean, shared_std = self.apply_poe(shared_mean_m1, shared_lv_m1, shared_mean_m2, shared_lv_m2)

        shared_dist = self._post_dist(shared_mean, shared_std)
        shared = shared_dist.rsample()
        
        priv_m1_std = torch.exp(0.5 * priv_lv_m1)
        shared_m1_std = torch.exp(0.5 * shared_lv_m1)
        priv_m1_dist = self._post_dist(priv_mean_m1, priv_m1_std)
        priv_m1 = priv_m1_dist.rsample()
        shared_m1_dist = self._post_dist(shared_mean_m1, shared_m1_std)
        shared_m1 = shared_m1_dist.rsample()
        
        priv_m2_std = torch.exp(0.5 * priv_lv_m2)
        shared_m2_std = torch.exp(0.5 * shared_lv_m2)
        priv_m2_dist = self._post_dist(priv_mean_m2, priv_m2_std)
        priv_m2 = priv_m2_dist.rsample()
        shared_m2_dist = self._post_dist(shared_mean_m2, shared_m2_std)
        shared_m2 = shared_m2_dist.rsample()

        # M1 + shared M1
        rec_m1 = self._decoder1(torch.cat((priv_m1, shared_m1), dim=1))
        if self._mask1 is not None:
            rec_m1[:, ~self._mask1] = 0.0
        
        #Private B + shared B
        rec_m2 = self._decoder2(torch.cat((priv_m2, shared_m2), dim=1))
        if self._mask2 is not None:
            rec_m2[:, ~self._mask2] = 0.0

        #Private A + PoE
        rec_m1_shared = self._decoder1(torch.cat((priv_m1, shared), dim=1))
        if self._mask1 is not None:
            rec_m1_shared[:, ~self._mask1] = 0.0
        
        #Private B + PoE
        rec_m2_shared = self._decoder2(torch.cat((priv_m2, shared), dim=1))
        if self._mask2 is not None:
            rec_m2_shared[:, ~self._mask2] = 0.0

        #Prior A + Shared B -> A
        prior_1 = self._post_dist(
            torch.zeros((batch_size, self._encoder1.private_size),
                        device=m1.device),
            torch.ones((batch_size, self._encoder1.private_size),
                       device=m1.device)).rsample()
        rec_m1_m2 = self._decoder1(torch.cat((prior_1, shared_m2), dim=1))
        if self._mask1 is not None:
            rec_m1_m2[:, ~self._mask1] = 0.0

        #Prior B + Shared A -> B
        prior_2 = self._post_dist(
            torch.zeros((batch_size, self._encoder2.private_size),
                        device=m2.device),
            torch.ones((batch_size, self._encoder2.private_size),
                       device=m2.device)).rsample()
        rec_m2_m1 = self._decoder2(torch.cat((prior_2, shared_m1), dim=1))
        if self._mask2 is not None:
            rec_m2_shared[:, ~self._mask2] = 0.0
        
        model_output = {'dists': [shared_dist, shared_m1_dist, shared_m2_dist,
                                  priv_m1_dist, priv_m2_dist],
                        'recs': [(rec_m1, rec_m2),
                                 (rec_m1_shared, rec_m2_shared),
                                 (rec_m1_m2, rec_m2_m1)],
                        'shared': shared_mean,
                        'priv_m1': priv_mean_m1,
                        'priv_m2': priv_mean_m2,
                        'stds': (shared_std.mean(), shared_m1_std.mean(), shared_m2_std.mean()),
                        'masks': (self._mask1, self._mask2)}

        return model_output

    def _encode(self, loader):
        dataset = loader.dataset
        data = torch.zeros((len(dataset),
                            self._input_shape[0],
                            self.latent_dimension),
                           requires_grad=False)
        targets = torch.zeros((len(dataset,)),
                              requires_grad=False)
        start_ix = 0
        for (i, batch) in enumerate(loader):
            x, y = batch
            num_subjects = x.size(0)
            end_ix = start_ix + num_subjects

            x = x.view(-1, *self._input_shape[1:]).float()
            with torch.no_grad():
                mu, _ = self._encoder(x)
                data[start_ix:end_ix] = mu.view(num_subjects, -1,
                                                self._latent_dimension)

            targets[start_ix:end_ix] = y.squeeze()

            start_ix = end_ix

        return data, targets

    def decode_dims(self, z):
        rec_m1 = self._decoder1(z)
        rec_m2 = self._decoder2(z)
        rec_m1[:, ~self._mask] = 0.0
        return rec_m1, rec_m2

    def interpolate_shared(self, latent_ix, minmax, avg, num_steps=5):
        priv_m1_avg, priv_m2_avg = avg
        priv_m1_avg = priv_m1_avg.unsqueeze(0).repeat(num_steps, 1)
        priv_m2_avg = priv_m2_avg.unsqueeze(0).repeat(num_steps, 1)
        int_m1_ls, int_m2_ls = [], []
        for (i, ix) in enumerate(latent_ix):
            z = torch.zeros((num_steps, self._encoder1.shared_size), device=minmax.device)
            z[:, ix] = torch.linspace(float(minmax[ix, 0]), float(minmax[ix, 1]), steps=num_steps)
            int_m1 = self._decoder1(torch.cat((priv_m1_avg, z), dim=1))
            int_m2 = self._decoder2(torch.cat((z, priv_m2_avg), dim=1))
            if self._mask1 is not None:
                int_m1[:, ~self._mask1] = 0.0
            if self._mask2 is not None:
                int_m2[:, ~self._mask2] = 0.0
            int_m1_ls.append(int_m1)
            int_m2_ls.append(int_m2)
        int_m1s = torch.stack(int_m1_ls, dim=0)
        int_m2s = torch.stack(int_m2_ls, dim=0)
        return int_m1s, int_m2s

    def reconstruct_cluster(self, cluster):
        with torch.no_grad():
            rec_m1 = self._decoder1(cluster[:, :(self._encoder1.private_size + self._encoder1.shared_size)])
            rec_m2 = self._decoder2(cluster[:, self._encoder1.private_size:])
            if self._mask1 is not None:
                rec_m1[:, ~self._mask1] = 0.0
            if self._mask2 is not None:
                rec_m2[:, ~self._mask2] = 0.0
        return rec_m1, rec_m2

    def cross_reconstruction(self, m1, m2):
        batch_size = m1.size(0)

        m1 = self._act(self._bn1(self._encoder1(m1)))
        m2 = self._act(self._bn2(self._encoder2(m2)))

        shared_m1, priv_m1 = self.split_dims(m1)
        shared_m2, priv_m2 = self.split_dims(m2)
        
        priv_mean_m1, priv_lv_m1 = self._priv_m1_mean(priv_m1), self._priv_m1_lv(priv_m1).clamp(EPS, MAX_LOG_VAR)
        priv_mean_m2, priv_lv_m2 = self._priv_m2_mean(priv_m2), self._priv_m2_lv(priv_m2).clamp(EPS, MAX_LOG_VAR)
        shared_mean_m1, shared_lv_m1 = self._s_m1_mean(shared_m1), self._s_m1_lv(shared_m1).clamp(EPS, MAX_LOG_VAR)
        shared_mean_m2, shared_lv_m2 = self._s_m2_mean(shared_m2), self._s_m2_lv(shared_m2).clamp(EPS, MAX_LOG_VAR)
    
        prior_m1 = self._post_dist(torch.zeros((batch_size, self._encoder1.private_size), device=m1.device),
                                   torch.ones((batch_size, self._encoder1.private_size), device=m1.device))
        prior_m2 = self._post_dist(torch.zeros((batch_size, self._encoder2.private_size), device=m2.device),
                                   torch.ones((batch_size, self._encoder2.private_size), device=m2.device))
        
        
        m1_z = torch.cat((priv_mean_m1, shared_mean_m1), dim=1)
        m2_z = torch.cat((priv_mean_m2, shared_mean_m2), dim=1)
        m1_s = torch.zeros((batch_size, self._encoder1_size), device=m1.device)
        m2_s = torch.zeros((batch_size, self._encoder2_size), device=m1.device)
        m1_m2 = torch.cat((torch.zeros((batch_size, self._encoder1.private_size), device=m1.device)
                           ,shared_mean_m2), dim=1)
        m2_m1 =  m1_m2 = torch.cat((torch.zeros((batch_size, self._encoder1.private_size), device=m1.device)
                           ,shared_mean_m1), dim=1)
        rec_m1_z = self._decoder1(m1_z)
        rec_m2_z = self._decoder2(m2_z)
        if self._mask1 is not None:
            rec_m1_z[:, ~self._mask1] = 0.0
        if self._mask2 is not None:
            rec_m2_z[:, ~self._mask2] = 0.0
        
        rec_m1_s = self._decoder1(m1_s)
        rec_m2_s = self._decoder2(m2_s)
        if self._mask1 is not None:
            rec_m1_s[:, ~self._mask1] = 0.0
        if self._mask2 is not None:
            rec_m2_s[:, ~self._mask2] = 0.0
        
        rec_m1_c = self._decoder1(m1_m2)
        rec_m2_c = self._decoder2(m2_m1)

        if self._mask1 is not None:
            rec_m1_c[:, ~self._mask1] = 0.0
        if self._mask2 is not None:
            rec_m2_c[:, ~self._mask2] = 0.0
        
        out_dict = {
            'rec_z': (rec_m1_z, rec_m2_z),
            'rec_s': (rec_m1_s, rec_m2_s),
            'rec_c': (rec_m1_c, rec_m2_c),
            'masks': (self._mask1, self._mask2)
        }
        return out_dict


    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Norm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m.bias, 'data'):  
                nn.init.constant_(m.bias.data, 0)

    def __str__(self):
        return 'DMVAE'
