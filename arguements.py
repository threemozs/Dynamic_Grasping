import argparse
import torch

# define some arguments that will be used...
def achieve_args():

    parse = argparse.ArgumentParser()

    parse.add_argument('--model_name', type=str, default='params007_', help='model name') #AntPyBulletEnv-v0
    parse.add_argument('--noise_in_sampling', type=str, default='Y', help='if add noise in sampling') #AntPyBulletEnv-v0

    parse.add_argument('--lr', type=float, default=1e-4, help='change to this after iter 250')
    # parse.add_argument('--lr', type=float, default=5e-5, help='change to this after iter 170')
    # parse.add_argument('--lr', type=float, default=1e-4, help='the learning rate of actor network')

    parse.add_argument('--gamma', type=float, default=0.995, help='the discount ratio...')
    parse.add_argument('--epsilon', type=float, default=0.2, help='the clipped ratio...')
    parse.add_argument('--tau', type=float, default=0.97, help='the coefficient for calculate GAE')


    parse.add_argument('--sample_traj_length', type=int, default=1000, help='sample trajectory number')
    parse.add_argument('--sample_point_num', type=int, default=256, help='sample point number')

    parse.add_argument('--epoch_num', type=int, default=10, help='epoch number for each sampled data ')
    parse.add_argument('--optim_batchsz', type=int, default=64, help='epoch number for each sampled data ')

    # REWARD-------
    parse.add_argument('--fall_penalty', type=int, default=-20, help='fall_penalty')
    parse.add_argument('--succcess_reward', type=int, default=1000, help='succcess_reward')

    args = parse.parse_args()

    return args

# increase pos shift threshold,
# add stay on reward

# check the noise
# reduce noise

# reduce learning rate

