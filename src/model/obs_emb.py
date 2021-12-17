import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ObsEncode(nn.Module):
    """
    Observation encode network
    """
    def __init__(self, use_relation=False, latent_size=0):
        super(ObsEncode, self).__init__()
        self.per_emb_dim = 32  # 64
        self.head = 6  # 8
        self.use_relation = use_relation
        if latent_size != 0:
            self.key_self = nn.Sequential(
                nn.Linear(22, self.per_emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.per_emb_dim, 16 * self.head)
            )
            self.query_others = nn.Sequential(
                nn.Linear(22, self.per_emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.per_emb_dim, 16 * self.head)
            )
        else:
            self.key_self = nn.Sequential(
                nn.Linear(22, self.per_emb_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.per_emb_dim // 2, self.per_emb_dim)
            )
            self.query_others = nn.Sequential(
                nn.Linear(22, self.per_emb_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.per_emb_dim // 2, self.per_emb_dim)
            )
        self.value_others = nn.Sequential(
            nn.Linear(22, 32),  # 32
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)  # 64
        )
        self.query_landmark = nn.Sequential(
            nn.Linear(2, self.per_emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.per_emb_dim // 2, self.per_emb_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16),   # 48->32
            nn.ReLU(),
            nn.Linear(16, 64)  # 32->32
        )
        self.reset_para()
    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs, task, relation=None, skill=None):
        bs = obs.size(0)  # obs: bs,n_agents,s
        n_a = obs.size(1)
        obs = obs.view(-1, obs.size(2))  # bs*n,s
        obs_self, obs_others, obs_landmark, n_other, n_l = devide_obs(obs, task)
        k_self = self.key_self(obs_self[:, None, :])
        q_others = self.query_others(obs_others)
        v_others = self.value_others(obs_others)  # bs*n,3,-1
        if skill is not None:  # bs,n,latent_size, relation network
            k_self = k_self.view(k_self.size(0), k_self.size(1), self.head, -1)
            q_others = q_others.view(q_others.size(0), q_others.size(1), self.head, -1)  # bs*n,3,4,-1
            skill = skill.view(-1, 1, skill.size(2), 1)
            k_self = k_self * skill  # bs*n,1,4,-1 x bs*n,1,4,1
            k_self = torch.sum(k_self, 2)  # bs*n,1,-1
            q_others = q_others * skill  # bs*n,3,4,-1 x bs*n,1,4,1
            q_others = torch.sum(q_others, 2)  # bs*n,3,-1 
        if not self.use_relation:
            others = F.softmax(1 / math.sqrt(2 * self.per_emb_dim) * torch.matmul(k_self, q_others.permute(0,2,1)), -1)
            relation = others.view(bs, -1, others.size(-1))   # bs,n,n_others
        else:
            # relation shape: num_agents_same_species,bs,num_agents
            others = relation.permute(1, 0, 2).contiguous().view(relation.size(0) * relation.size(1), 1, -1)  # bs*n,1,3
        others = torch.matmul(others, v_others).squeeze().view(bs, n_a, -1)  # (bs*n,1,n_ot)x(bs*n,n_ot,-1)=bs*n,1,-1
        return others.permute(0, 2, 1), relation


def devide_obs(obs, task):
    # per task: [num_agents, [num_good_agents, num_adversaries, num_landmarks]]
    others_num = task[0] - 1
    landmark_num = task[1][-1]
    self_dim = 4
    per_other_dim = 22  # 36
    per_lm_dim = 3
    obs_self = obs[:, :self_dim]
    obs_others = obs[:, self_dim: self_dim + per_other_dim * others_num]
    obs_landmark = obs[:, self_dim + per_other_dim * others_num: self_dim + per_other_dim * others_num + per_lm_dim* landmark_num]
    obs_self = torch.cat((obs_self, obs_landmark), -1)
    obs_others = obs_others.view(obs_others.size(0), others_num, per_other_dim)
    return obs_self, obs_others, obs_landmark, others_num, landmark_num
