import numpy as np
import torch
from utils.misc import reset_skill
import math


def relation_reward_deterministic(target_act, all_probs, all_prob_samples, num_samples):
    # for marginalization of the denominator
    target_act = target_act.max(2, True)[1].detach()  # n_agents,bs,1
    target_act_prob = all_probs.gather(2, target_act)  # n_agents,bs,1
    target_act_prob_samples = all_prob_samples.gather(2, target_act[:, :, None, :].repeat(1, 1, num_samples, 1).view(target_act.size(0), -1, 1))  # n_agents,bs*samples,1
#    if all_prob_samples.size(0) == 10:
    if True:
        print(all_prob_samples.view(-1, 1024, 5, 5)[0,0,:,:], "all_prob_samples", all_prob_samples.size())
    #return s
   # print(target_act_prob[0, 0, :],  target_act_prob_samples.view(target_act_prob.size(0), target_act_prob.size(1),
   #                                                        -1, 1)[0, 0, :,:], target_act[0, 0, :])
    target_act_prob_samples = target_act_prob_samples.view(target_act_prob.size(0), target_act_prob.size(1),
                                                           -1, 1).sum(2)  # n,bs,samples,1->n,bs,1
    reward = torch.log(num_samples * target_act_prob).sum(0) - torch.log(target_act_prob_samples).sum(0)
#    reward = torch.log(num_samples * target_act_prob).mean(0) - torch.log(target_act_prob_samples).mean(0)
    print(reward.mean(), "reward")
    return reward.mean()
