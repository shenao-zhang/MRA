import torch
import torch.nn as nn
from src.model.obs_emb import ObsEncode


class CriticValue(nn.Module):
    def __init__(self, act_dim):
        super(CriticValue, self).__init__()
        self.obs_encode = ObsEncode(use_relation=False)
        self.hidden_dim = 64
        self.mid_dim = 32
        self.critic_v = nn.Sequential(
            nn.Conv1d(self.hidden_dim + act_dim, self.mid_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.mid_dim, self.mid_dim // 2, 1)
        )
        self.critic_v_role2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim + act_dim, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, act_dim, 1)
        )
        self.self_v = nn.Sequential(
            nn.Conv1d(self.hidden_dim + act_dim, self.mid_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.mid_dim, self.mid_dim // 2, 1)
        )
        self.post_fc = nn.Sequential(
            nn.Conv1d(self.mid_dim, act_dim, 1)
        )
        

    def forward(self, task, obs, actions, relations, agent_divide=None, return_all_q=False, logger=None, niter=0):
        obs, _ = self.obs_encode(obs, task, relations)  # bs,64,n_agents
        act = actions.permute(1, 2, 0).contiguous()
        sa_emb = torch.cat((obs.view(obs.size(0), -1, obs.size(-1)), act), 1)  # bs,64+dim_act,n_agents
        assert sa_emb.size(-1) == agent_divide[0]
        role1_emb = sa_emb[:, :, :agent_divide[0]]
        critic_val = self.critic_v(role1_emb)
        critic_val = critic_val.view(critic_val.size(0), -1, critic_val.size(-1))  # bs,-1,n
        rel = []
        relations = relations.view(relations.size(0), relations.size(1), -1)  # n,bs,n_ot
        for n_a in range(relations.size(0)):
            tem = torch.cat((relations[n_a, :, :n_a], torch.ones(relations.size(1))[:, None].cuda(), relations[n_a, :, n_a:]), -1)
            rel.append(tem)
        rel = torch.stack(rel, 0).permute(1, 2, 0)  # bs,n_rel,n
        self_v = self.self_v(role1_emb).view(rel.size(0), -1, rel.size(-1))  # bs,-1,n
        critic_val = torch.matmul(critic_val, rel)  # the order is important here, (bs,-1,n)
        critic_val = torch.cat((critic_val, self_v), 1)  # bs,2*(-1),n
        critic_val = self.post_fc(critic_val).permute(2, 0, 1)   # n,bs,act
        action_taken = actions.max(dim=2, keepdim=True)[1]
        q = critic_val.gather(2, action_taken)  # n_agents,bs,1
        for agent_id in range(q.size(0)):
            r = relations[agent_id, :, :]  # bs,n
            relation_entropy = -((r+0.00001).log() * r).sum(-1)
            if logger is not None:
                logger.add_scalar('agent%i/relation_entropy' % agent_id, relation_entropy.mean(), niter)
        if return_all_q:
            return q, critic_val
        else:
            return q

