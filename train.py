import torch
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from src.relation_marl import RelationalMARL
from utils.misc import *
from statistics import mean


def make_parallel_env(env_id, n_rollout_threads, seed, task):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, task, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    torch.manual_seed(run_num)
    np.random.seed(run_num)
    
    model = RelationalMARL.init_from_env(tau=config.tau,
                                         pi_lr=config.pi_lr,
                                         q_lr=config.q_lr,
                                         gamma=config.gamma,
                                         reward_scale=config.reward_scale,
                                         latent_size=config.skill_num,
                                         act_size=config.act_num,
                                         resample_num=config.resample_num,
                                         species=config.species,
                                         num_tasks=len(config.task_list))
    task_count = [0] * len(config.task_list)
    logger_list = []
    env_list = []
    buffer_list = []
    for task in config.task_list:
        log_dir = run_dir / str(task[1]) / 'logs'   # task[0]:n_agents, task[1]: (n_agent_good, n_agent_adv,n_landmark)
        os.makedirs(log_dir)
        logger = SummaryWriter(str(log_dir))
        logger_list.append(logger)
        env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num, task)
        assert task[0] == len(env.observation_space)
        env_list.append(env)
        replay_buffer = ReplayBuffer(config.buffer_length, task[0],
                                     [obsp.shape[0] for obsp in env.observation_space],  # nagents,13
                                     [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                      for acsp in env.action_space])
        buffer_list.append(replay_buffer)

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        for task_id, task in enumerate(config.task_list):
            print("Episodes %i-%i of %i, task is %s" % (ep_i + 1, ep_i + 1 + config.n_rollout_threads,
                                                        config.n_episodes, str(task)))
            env = env_list[task_id]
            logger = logger_list[task_id]
            replay_buffer = buffer_list[task_id]
            obs = env.reset()  # bs,n_agents,-1
            model.prep_rollouts(device='cpu')
            latent_skill = model.latent_sample(task)
            for et_i in range(config.episode_length):
                torch_obs = Variable(torch.Tensor(obs[:, :, :]), requires_grad=False)  # bs,n_agents,-1
                torch_relations = model.relation_step(task, torch_obs, latent_skill).permute(1, 0, 2)  # n,bs,n
                relations = [rel.data.numpy() for rel in torch_relations]
                torch_agent_actions = model.step(task, torch_obs, torch_relations, config.spec_split[task_id],
                                                 explore=True)
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                # rearrange actions to be per environment
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                next_obs, rewards, dones, infos = env.step(actions)
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones, relations)
                obs = next_obs
                task_count[task_id] += config.n_rollout_threads
                if (len(replay_buffer) >= config.batch_size and (task_count[task_id] % config.steps_per_update) < config.n_rollout_threads):
                    if config.use_gpu:
                        model.prep_training(device='gpu')
                    else:
                        model.prep_training(device='cpu')
                    for u_i in range(config.num_updates):
                        sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                        model.update_critic(task, sample, spec_split=config.spec_split[task_id], logger=logger)
                        model.update_policies(task, sample, spec_split=config.spec_split[task_id], logger=logger)
                    model.update_relation_net(task, sample, spec_split=config.spec_split[task_id],
                                              all_tasks=config.task_list, logger=logger)
                    model.update_all_targets()
                    model.update_task_net(task, sample, task_id, logger)
                    model.prep_rollouts(device='cpu')
            ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
            for a_i, a_ep_rew in enumerate(ep_rews):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            logger.add_scalar('rewards/mean_episode_rewards', mean(ep_rews), ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    for env, logger, task in zip(env_list, logger_list, config.task_list):
        env.close()
        log_dir = run_dir / str(task[1]) / 'logs'
        logger.export_scalars_to_json(str(log_dir / 'summary.json'))
        logger.close()

