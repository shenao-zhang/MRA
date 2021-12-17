import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample
from src.model.obs_emb import ObsEncode


class BasePolicy(nn.Module):
    def __init__(self, act_dim):
        super(BasePolicy, self).__init__()
        self.obs_encode = ObsEncode(use_relation=True)
        self.hidden_dim = 64
        self.fc_act = nn.Sequential(
            nn.Conv1d(self.hidden_dim, 32, 1),   # 32
            nn.ReLU(inplace=True),
            nn.Conv1d(32, act_dim, 1)  # 32
        )
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0, 0.1)

    def forward(self, obs, relation, agent_id, task):
        obs_emb, _ = self.obs_encode(obs, task, relation)  # bs,64,n
        out = self.fc_act(obs_emb)  # bs,act,n
        return out


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, task, obs, relation, agent_id, explore=True, return_all_probs=False,
                return_log_pi=False, regularize=False, return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs, relation, agent_id, task)   # bs,act,n
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if explore:
            probs_sample = probs.permute(0, 2, 1).contiguous().view(-1, out.size(1))
            int_act, act = categorical_sample(probs_sample, use_cuda=on_gpu)
            int_act = int_act.view(out.size(0), out.size(2), 1).permute(0, 2, 1).contiguous()
            act = act.view(out.size(0), out.size(2), out.size(1)).permute(0, 2, 1)  # bs,act,n
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append((out**2).mean(1))
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
