import torch
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
from utils.misc import *
from src.agents import MRAgents
from src.model.critics_value import CriticValue
from src.relation_rewards import relation_reward_deterministic
from src.model.relation_net import Relation
from src.task_net import TaskEncode, TaskPred


class RelationalMARL(object):
    def __init__(self, latent_size, act_size, resample_num,
                 gamma=0.95, tau=0.01, pi_lr=0.001, q_lr=0.001,
                 reward_scale=10., species=2, num_tasks=8, **kwargs):
        """
        Inputs:
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal policy entropy)
        """
        self.species = species
        self.nagents = None
        self.resample_num = resample_num
        self.task_encode = TaskEncode(latent_dim=latent_size)
        self.task_predictor = TaskPred(latent_dim=latent_size, num_tasks=num_tasks)
        self.task_net_para = [{"params": self.task_encode.parameters()}, {"params": self.task_predictor.parameters()}]
        self.task_optimizer = Adam(self.task_net_para, lr=q_lr, weight_decay=3e-4)
        self.agents = [MRAgents(action_size=act_size)
                       for spe_id in range(self.species)]
        self.critic = CriticValue(act_dim=act_size)
        self.target_critic = CriticValue(act_dim=act_size)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=3e-4)  # 3
        self.rel_net = Relation(latent_dim=latent_size, resample_num=resample_num)
        self.rel_task_para = [{"params": self.rel_net.parameters()}]
        self.rel_optimizer = Adam(self.rel_task_para, lr=pi_lr, weight_decay=3e-4)
        self.pol_para = [{"params": a.policy.parameters()} for a in self.agents]
        self.pol_optimizer = Adam(self.pol_para, lr=pi_lr, weight_decay=3e-4)
        self.latent_size = latent_size
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.MSELoss = torch.nn.MSELoss()
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.relation_net_dev = 'cpu'  # device for relation net
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, task, observations, relations, spec_split, explore=True, return_log_pi=False, target_pi=False,
             return_all_probs=False, regularize=False, return_entropy=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        action = []
        log_pi = []
        probs_all = []
        reg_all = []
        entropy_all = []
        if return_entropy:   # return_entropy is true only when update policies
            for n_spe, spe_id in enumerate(spec_split):
                obs = observations[:, spe_id, :]  # bs,n_spe,-1
                rel = relations[spe_id, :, :]  # n_spe,bs,n_agents
                act, probs, l_pi, reg, ent = self.agents[n_spe].step(task, obs, rel, agent_id=spe_id, explore=explore,
                                                                     return_log_pi=return_log_pi, target_pi=target_pi,
                                                                     return_all_probs=return_all_probs,
                                                                     regularize=regularize,
                                                                     return_entropy=return_entropy)  # bs,act,n_spe
                action.append(act)
                probs_all.append(probs)
                log_pi.append(l_pi)
                reg_all.append(reg)
                entropy_all.append(ent)
            action = torch.cat(action, -1)
            probs_all = torch.cat(probs_all, -1)
            log_pi = torch.cat(log_pi, -1)
            reg_all = torch.cat(reg_all, -1)
            self.nagents = action.size(-1)
            return action.permute(2, 0, 1), probs_all.permute(2, 0, 1), log_pi.permute(2, 0, 1),\
                   reg_all, entropy_all
        elif return_all_probs and target_pi:
            for n_spe, spe_id in enumerate(spec_split):
                obs = observations[:, spe_id, :]  # bs,n_spe,-1
                rel = relations[spe_id, :, :]  # n_spe,bs,n_agents
                act, probs = self.agents[n_spe].step(task, obs, rel, agent_id=spe_id, explore=explore,
                                                                     return_log_pi=return_log_pi, target_pi=target_pi,
                                                                     return_all_probs=return_all_probs,
                                                                     regularize=regularize,
                                                                     return_entropy=return_entropy)  # bs,act,n_spe
                action.append(act)
                probs_all.append(probs)
            action = torch.cat(action, -1)
            probs_all = torch.cat(probs_all, -1)
            self.nagents = action.size(-1)
            return action.permute(2, 0, 1), probs_all.permute(2, 0, 1)
        elif return_log_pi:
            for n_spe, spe_id in enumerate(spec_split):
                obs = observations[:, spe_id, :]
                rel = relations[spe_id, :, :]
                act, l_pi = self.agents[n_spe].step(task, obs, rel, agent_id=spe_id, explore=explore,
                                                    return_log_pi=True, target_pi=target_pi)  # bs,act,n_spe
                action.append(act)
                log_pi.append(l_pi)
            action = torch.cat(action, -1)
            log_pi = torch.cat(log_pi, -1)
            self.nagents = action.size(-1)
            return action.permute(2, 0, 1), log_pi.permute(2, 0, 1)  # nagents,bs,act; nagents,bs,1
        else:
            for n_spe, spe_id in enumerate(spec_split):
                obs = observations[:, spe_id, :]
                rel = relations[spe_id, :, :]  # n_spe,bs,n_agents
                act = self.agents[n_spe].step(task, obs, rel, agent_id=spe_id, explore=explore, return_log_pi=False,
                                              target_pi=target_pi)  # bs,act,n_spe
                action.append(act)
            action = torch.cat(action, -1)
            self.nagents = action.size(-1)
            return action.permute(2, 0, 1)  # nagents,bs,act

    def relation_step(self, task, observations, latent_skill, calculate_reward=False, task_emb=None, all_tasks=None):
        a = self.rel_net.forward(task, observations, latent_skill, calculate_reward, task_emb, all_tasks)
        return a

    def latent_sample(self, task):
        num_agents = task[0]
        z_samples = self.task_encode(task)  # 1,latent_size
        z_samples = z_samples.repeat(num_agents, 1)  # n,latent_size
        z_samples = gumbel_softmax(z_samples, hard=True, temperature=1, dim=-1)  # n,latent_size
        return z_samples


    def update_critic(self, task, sample, spec_split, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, act, rews, next_obs, dones, relations = sample
        obs = torch.stack(obs, 0).permute(1, 0, 2).contiguous()
        next_obs = torch.stack(next_obs, 0).permute(1, 0, 2).contiguous()
        relations = torch.stack(relations, 0)
        next_act, next_log_pi = self.step(task, obs, relations, spec_split, return_log_pi=True, target_pi=True)
        next_qs = self.target_critic(task, next_obs, next_act, relations, agent_divide=task[1])  # assume one step action not change relationships
        act = torch.stack(act, 0)
        critic_rets = self.critic(task, obs, act, relations, agent_divide=task[1], logger=logger, niter=self.niter)
        rews = torch.stack(rews, 0).view(self.nagents, -1, 1)  # nagents,bs,1
        dones = torch.stack(dones, 0).view(self.nagents, -1, 1)  # nagents,bs,1
        target_q = rews + self.gamma * next_qs * (1 - dones)  # nagents,bs,1
        if soft:
            target_q -= next_log_pi / self.reward_scale
        q_loss = self.MSELoss(critic_rets.sum(0), target_q.sum(0).detach())  # calculate q loss (bs,1)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, task, sample, spec_split, soft=True, regularize=False, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones, relations = sample
        obs = torch.stack(obs, 0).permute(1, 0, 2).contiguous()
        relations = torch.stack(relations, 0)
        curr_act, all_probs, all_log_pi, pol_regs, ent = self.step(task, obs, relations, spec_split,
                                                                   return_all_probs=True, return_log_pi=True,
                                                                   regularize=True, return_entropy=True)
        q, all_q = self.critic(task, obs, curr_act, relations, agent_divide=task[1], return_all_q=True)
        pol_target = q   # nagents,bs,1
        if soft:
            pol_loss = (all_log_pi * (all_log_pi / self.reward_scale - pol_target).detach()).sum(0).mean()
        else:
            pol_loss = (all_log_pi * (-pol_target).detach()).sum(0).mean()   # (nagents,bs,1)->(bs,1)->1
        if logger is not None:
            logger.add_scalar('losses/pol_loss', pol_loss, self.niter)
            logger.add_scalar('entropy/pol_entropy', sum(ent), self.niter)
        self.pol_optimizer.zero_grad()
        pol_loss.backward()
        self.pol_optimizer.step()
        self.pol_optimizer.zero_grad()

    def update_relation_net(self, task, sample, spec_split, all_tasks, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones, _ = sample
        obs = torch.stack(obs, 0).permute(1, 0, 2).contiguous()   # bs,n_agents,-1
        self.task_encode.train()
        self.task_predictor.train()
        self.task_encode = self.task_encode.cuda()
        self.task_predictor = self.task_predictor.cuda()
        skill = self.latent_sample(task)  # n_agents,latent_size
        task_emb = self.task_encode(task)
        print("task:", task)
        print("task emb:", torch.exp(task_emb))
        relation, relation_samples, num_samples = self.relation_step(task, obs, skill, calculate_reward=True,
                                                                     task_emb=task_emb, all_tasks=all_tasks)  # bs,n,n
        relation = relation.permute(1, 0, 2).contiguous()  # n,bs,n
        relation_samples = relation_samples.permute(1, 0, 2).contiguous()
        
        act_ignore, all_probs = self.step(task, obs, relation, spec_split, target_pi=True,
                                          return_all_probs=True)  # target policy network
        obs_samples = obs[:, None, :, :].repeat(1, num_samples, 1, 1).view(-1, obs.size(1), obs.size(2))
        act_ignore, all_prob_samples = self.step(task, obs_samples, relation_samples, spec_split, target_pi=True,
                                                 return_all_probs=True)  # target policy network, relation resample 
        target_act = self.step(task, obs, relation, spec_split, explore=True)  # current policy act as desired act
        target_act = target_act.detach()  # we only want to optimize relation network here
        rel_rewards = relation_reward_deterministic(target_act, all_probs, all_prob_samples, num_samples)
        rel_loss = -rel_rewards
        self.rel_optimizer.zero_grad()
        rel_loss.backward()
        print("MIgrad", torch.nn.utils.clip_grad_norm(self.rel_net.parameters(), 10 * self.nagents))
        grad = torch.nn.utils.clip_grad_norm(self.rel_net.parameters(), 10 * self.nagents)
        self.rel_optimizer.step()
        self.rel_optimizer.zero_grad()
        self.task_encode.eval()
        self.task_predictor.eval()
        self.task_encode = self.task_encode.cpu()
        self.task_predictor = self.task_predictor.cpu()
        if logger is not None:
            logger.add_scalar('losses/relation_loss', rel_loss, self.niter)
            logger.add_scalar('grad_norms/relation_net', grad, self.niter)

    def update_task_net(self, task, sample, task_id, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones, _ = sample
        obs = torch.stack(obs, 0).permute(1, 0, 2).contiguous()  # bs,n_agents,-1
        self.task_encode.train()
        self.task_predictor.train()
        self.task_encode = self.task_encode.cuda()
        self.task_predictor = self.task_predictor.cuda()
        skill = self.latent_sample(task)  # n_agents,latent_size
        relation = self.relation_step(task, obs, skill)  # bs,n,n
        task_pred = self.task_predictor(obs, relation, task, skill)  # predict a task prob distribution, bs,n_tasks
        task_label = Variable(torch.Tensor([task_id]), requires_grad=False)[None, :].repeat(task_pred.size(0), 1).cuda().squeeze()
        task_label = task_label.type(torch.long)
        pred_loss = self.CELoss(task_pred, task_label)
        task_loss = pred_loss 
        self.task_optimizer.zero_grad()
        task_loss.backward()
        self.task_optimizer.step()
        self.task_optimizer.zero_grad()
        
        task_pred = F.softmax(task_pred, -1)
        self.task_predictor.eval()
        self.task_encode.eval()
        self.task_predictor = self.task_predictor.cpu()
        self.task_encode = self.task_encode.cpu()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        hard_update(self.target_critic, self.critic)
        for a in self.agents:
            hard_update(a.target_policy, a.policy)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        self.rel_net.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device
        if not self.relation_net_dev == device:
            self.rel_net = fn(self.rel_net)
            self.relation_net_dev = device

    def prep_rollouts(self, device='cpu'):
        self.rel_net.eval()
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.relation_net_dev == device:
            self.rel_net = fn(self.rel_net)
            self.relation_net_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'relation_params': self.rel_net.state_dict(),
                     'task_encoder_params': self.task_encode.state_dict(),
                     'rel_optimizer': self.rel_optimizer.state_dict(),
                     'pol_optimizer': self.pol_optimizer.state_dict(), 
                     'task_optimizer': self.task_optimizer.state_dict(),
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01, reward_scale=10.,
                      species=2, latent_size=4, act_size=5, resample_num=30, num_tasks=8,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'species': species, 'latent_size': latent_size,
                     'act_size': act_size, 'resample_num': resample_num, 'num_tasks': num_tasks}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=True):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        instance.task_encode.load_state_dict(save_dict['task_encoder_params'])
        instance.rel_net.load_state_dict(save_dict['relation_params'])
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
        return instance
