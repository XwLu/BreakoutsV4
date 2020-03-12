#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse

def parse(parser):
    parser.add_argument('--env_name', default='BreakoutNoFrameskip-v4', help='environment name')
    parser.add_argument('--n_obs', default=None, help='dim of observation')
    parser.add_argument('--n_action', default=None, help='dim of actions')
    parser.add_argument('--update_times', default=4, help='number of policy improvement')
    parser.add_argument('--update_freqence', default=4, help='update freqence of parameters.')
    parser.add_argument('--lr', default=1e-4, help='learning rate')
    parser.add_argument('--gamma', default=0.99, help='Attenuation coefficient of Q value')
    parser.add_argument('--max_grad_norm', default=0.5, help='max grad in optimization step')
    parser.add_argument('--num_procs', default=8, help='number of process')
    parser.add_argument('--buffer_size', default=5000, help='size of samples restored in replay buffer')
    parser.add_argument('--batch_size', default=40, help='size of training batch')
    parser.add_argument('--random_epoches', default=10000, help='collect experience with random policy')
    parser.add_argument('--train_epoches', default=300000, help='number of train epoches')
    parser.add_argument('--final_epsilon', default=0.1, help='final epsilon value of epsilon-greedy policy')
    parser.add_argument('--log_interval', default=100, help='log interval')
    parser.add_argument('--save_interval', default=1000, help='save interval')
    parser.add_argument('--test_interval', default=1000, help='test interval')
    parser.add_argument('--test_episode', default=10, help='test episode')    
    parser.add_argument('--update_target_interval', default=100, help='update target interval')
    
    args = parser.parse_args()
    return args