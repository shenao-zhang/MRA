import argparse
from train import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="food", help="Name of environment")
    parser.add_argument("--model_name", default="./test",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=80000, type=int)
    parser.add_argument("--episode_length", default=20, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)  # 100
    parser.add_argument("--num_updates", default=10, type=int,   # 4
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--pi_lr", default=0.0003, type=float)  # 0.001
    parser.add_argument("--q_lr", default=0.0003, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--reward_scale", default=30, type=float)

    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--skill_type", default='discrete_uniform', type=str,
                        choices=['discrete_uniform', 'gaussian', 'cont_uniform'])
    parser.add_argument("--skill_num", default=6, type=int)  # 8
    parser.add_argument("--resample_num", default=5, type=int)
    task_list = [[6, [6, 6], [1, 1, 2.5]], [9, [9, 6], [1, 1, 2.5]], [12, [12, 6], [1, 1, 2.5]]]
    parser.add_argument("--task_list", default=task_list)
    spec_list = [[[0,1,2,3,4,5]], [[0,1,2,3,4,5,6,7,8]], [[0,1,2,3,4,5,6,7,8,9,10,11]]]

    parser.add_argument("--spec_split", default=spec_list)
    parser.add_argument("--dim_entity", default=(4, 4, 2))
    parser.add_argument("--species", default=1, type=int)
    parser.add_argument("--act_num", default=5, type=int)
    config = parser.parse_args()
    run(config)
