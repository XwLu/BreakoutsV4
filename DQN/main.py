#!/usr/bin/env python
# -*- coding: utf-8 -*-


from arguments import parse
from agent import Agent, LinearSchedule
import sys
sys.path.append('..')
from common import make_env, print_dict
import argparse
import pretty_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Breakouts")
    parser.add_argument('--train', action='store_true', help='whether train DQN')
    parser.add_argument('--test', action='store_true', help='whether test DQN')
    args = parse(parser)

    env = make_env(args.env_name, 1000, args.num_procs)
    test_env = make_env(args.env_name, 1000, 1, clip_reward=False)
    args.n_obs = env.observation_space.shape[-1]
    args.n_action = env.action_space.n

    first_obs = env.reset()
    agent = Agent(args, first_obs)
    if args.train:
        agent.collect_experiences(env, args.random_epoches, 1.0) # collect some samples via random action
        epsilon_generator = LinearSchedule(args.train_epoches, args.final_epsilon)
        for i in range(args.train_epoches):
            epsilon = epsilon_generator(i)
            log = agent.collect_experiences(env, args.update_freqence, epsilon)
            info = agent.improve_policy(args.update_times)
            if i % args.log_interval == 0:
                print_dict({'step': i, 'epsilon': epsilon}, info, log)
            if i % args.save_interval == 0:
                print('Save Model')
                agent.save_weights()
            if i % args.update_target_interval == 0:
                agent.update_target_net()
            if i % args.test_interval == 0:
                print('=' * 20 + 'Test Agent' + '=' * 20)
                info = agent.evaluate(test_env)
                print_dict(info)
    elif args.test:
        agent.display(test_env)
    