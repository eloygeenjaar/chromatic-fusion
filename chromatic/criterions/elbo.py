from .basecriterion import BaseCriterion
import torch
from torch import distributions as D  # noqa
from torch import nn  # noqa


class ELBO(BaseCriterion):
    def __init__(self,
                 dataset: str,
                 beta: float,
                 prior_dist: D.Distribution,
                 use_mse_loss: bool):
        super().__init__()

        self._dataset = dataset
        self._beta = beta
        self._prior_dist = prior_dist
        self._kl_weight = 0.0
        self._use_mse_loss = use_mse_loss
        self._mse_loss = nn.MSELoss(reduction='none')

    def forward(self, m1, m2, model_results):
        m1 = m1.view(m1.size(0), -1)
        mask_m1, mask_m2 = model_results['masks']
        if mask_m2 is not None:
             m2 = m2.view(m2.size(0), -1)
        kl_shared = D.kl.kl_divergence(model_results['dists'][0], self._prior_dist).mean(-1)
        kl_s_m1 = D.kl.kl_divergence(model_results['dists'][1], self._prior_dist).mean(-1)
        kl_s_m2 = D.kl.kl_divergence(model_results['dists'][2], self._prior_dist).mean(-1)
        kl_priv_m1 = D.kl.kl_divergence(model_results['dists'][3], self._prior_dist).mean(-1)
        kl_priv_m2 = D.kl.kl_divergence(model_results['dists'][4], self._prior_dist).mean(-1)
        kl_loss = (kl_shared + kl_s_m1 + kl_s_m2 + kl_priv_m1 + kl_priv_m2) * self._beta
        if mask_m1 is not None:
            rec_m1 = self._mse_loss(model_results['recs'][0][0][:, mask_m1], m1[:, mask_m1]).mean(-1)
        else:
            rec_m1 = self._mse_loss(model_results['recs'][0][0], m1).mean(-1)
        if mask_m2 is not None:
            rec_m2 = self._mse_loss(model_results['recs'][0][1][:, mask_m2], m2[:, mask_m2]).mean(-1)
        else:
            rec_m2 = self._mse_loss(model_results['recs'][0][1], m2).mean(-1)
        rec_loss = (rec_m1 + rec_m2)
        for (m1_rec, m2_rec) in model_results['recs'][1:]:
            if mask_m1 is not None:
                rec_loss += self._mse_loss(m1_rec[:, mask_m1], m1[:, mask_m1]).mean(-1)
            else:
                rec_loss += self._mse_loss(m1_rec, m1).mean(-1)
            if mask_m2 is not None:
                rec_loss += self._mse_loss(m2_rec[:, mask_m2], m2[:, mask_m2]).mean(-1)
            else:
                rec_loss += self._mse_loss(m2_rec, m2).mean(-1)
        kl_loss /= 5
        rec_loss /= len(model_results['recs'])
        # The 500 multiplier ensures that the balance between the kl-loss and rec-loss is good for training
        loss = (100 * rec_loss + self._kl_weight * kl_loss).mean(0)
        # Linear KL-annealing
        self._kl_weight = min(self._kl_weight + 1/2000, 1)
        criterion_dict = {'loss': (100 * rec_loss + kl_loss).mean(0).detach(),
                          'rec': rec_loss.mean(0).detach(),
                          'kl_shared': kl_shared.mean(0).detach(),
                          'kl_s_m1': kl_s_m1.mean(0).detach(),
                          'kl_s_m2': kl_s_m2.mean(0).detach(),
                          'kl_p_m2': kl_priv_m2.mean(0).detach(),
                          'kl_p_m1': kl_priv_m1.mean(0).detach(),
                          'rec_m1': rec_m1.mean(0).detach(),
                          'rec_m2': rec_m2.mean(0).detach(),
                          'std_s': model_results['stds'][0].detach(),
                          'std_s1': model_results['stds'][1].detach(),
                          'std_s2': model_results['stds'][2].detach()}
        return loss, criterion_dict
    
    def __str__(self):
        return 'ELBO'

    @property
    def keys(self):
        return ['loss',
                'rec',
                'kl_shared',
                'kl_s_m1',
                'kl_p_m2',
                'kl_s_m2',
                'rec_m1',
                'rec_m2',
                'std_s', 'std_s1', 'std_s2']
