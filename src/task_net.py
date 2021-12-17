import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.model.obs_emb import ObsEncode


class TaskEncode(nn.Module):
    """
    Task encoder
    """
    def __init__(self, latent_dim):
        super(TaskEncode, self).__init__()

        self.hidden_dim = 8
        self.encoder = nn.Sequential(
            nn.Linear(1, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 3),
        )

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, task_des, calculate_loss=False):
        task_description = [task_des[0] / 10]
        task_description = Variable(torch.Tensor(task_description), requires_grad=False)
        task_description = task_description
        use_cuda = next(self.parameters()).is_cuda
        if use_cuda:
            task_description = task_description.cuda()
        task_emb = self.encoder(task_description)
        latent_pred = F.log_softmax(task_emb * 1 / 4, 0)  # 1,latent_size
        if not calculate_loss:
            return latent_pred
            

class TaskPred(nn.Module):
    """
    Task inference network
    """

    def __init__(self, latent_dim, num_tasks):
        super(TaskPred, self).__init__()

        self.hidden_dim = 64 + latent_dim
        self.obs_emb = ObsEncode(use_relation=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, num_tasks),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 3),
        )

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs, relation, task, skill):
        obs, _ = self.obs_emb(obs, task, relation)
        obs = obs.permute(0, 2, 1)
        obs = torch.cat((obs, skill[None, :, :].repeat(obs.size(0), 1, 1)), -1)
        emb = self.fc1(obs)  # bs,n,n_tasks
        emb = emb.sum(1)   # sum
        return emb
