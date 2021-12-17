from torch.optim import Adam
from utils.misc import hard_update
from src.model.policies import DiscretePolicy
from src.model.relation_net import Relation


class MRAgents(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, action_size):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = DiscretePolicy(action_size)
        self.target_policy = DiscretePolicy(action_size)

        hard_update(self.target_policy, self.policy)

    def step(self, task, obs, relation, agent_id, explore=True, return_log_pi=False, target_pi=False,
             return_all_probs=False, regularize=False, return_entropy=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if target_pi and not return_all_probs:  # used in critic update
            return self.target_policy(task, obs, relation, agent_id, explore=explore, return_log_pi=return_log_pi)
        elif target_pi and return_all_probs:  # used in relation net update
            return self.target_policy(task, obs, relation, agent_id, explore=explore, return_all_probs=return_all_probs)
        return self.policy(task, obs, relation, agent_id, explore=explore, return_log_pi=return_log_pi,
                           return_all_probs=return_all_probs, regularize=regularize, return_entropy=return_entropy)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict()}
        #        'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
       # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
