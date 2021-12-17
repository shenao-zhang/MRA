import torch.nn as nn
from src.model.obs_emb import ObsEncode
from utils.misc import *


class Relation(nn.Module):
    """
    Relationship network
    """
    def __init__(self, latent_dim, resample_num):
        """
        Inputs:
            state_dim (int): Number of dimensions in input state
            num_agents (int): Number of dimensions in output relationships
            latent_size(int): Dimension of latent space
            hidden_dim (int): Number of hidden dimensions
        """
        super(Relation, self).__init__()
        self.hidden_dim = 32
        self.resample_num = resample_num
        self.latent_dim = latent_dim
        self.obs_encode = ObsEncode(latent_size=latent_dim)

    def forward(self, task, obs, latent_skill, calculate_reward=False, task_emb=None, all_tasks=None):
        latent_skill = latent_skill[None, :, :].repeat(obs.size(0), 1, 1)  # bs,n,z
        obs_emb, rel = self.obs_encode(obs, task, None, latent_skill)  # bs,128,n_agents-1
        if not calculate_reward:
            return rel  # bs,n_agents,n_agents
        else:
            curr_task = task_emb[None, None, :].repeat(self.resample_num, obs.size(1), 1)  # resample,n_a,z_size
            z_samples = gumbel_softmax(curr_task, hard=True, temperature=1, dim=-1)  # resample,z_size
            skill = z_samples[None, :, :, :].repeat(obs.size(0), 1, 1, 1)  # bs,samples,n_a,z_size
            obs = obs[:, None, :, :].repeat(1, skill.size(1), 1, 1)  # bs,samples,n_a,-1
            skill = skill.view(skill.size(0) * skill.size(1), skill.size(2), skill.size(3))
            obs = obs.view(obs.size(0) * obs.size(1), obs.size(2), obs.size(3))
            _, relation_resample = self.obs_encode(obs, task, None, skill)
            return rel, relation_resample, self.resample_num

