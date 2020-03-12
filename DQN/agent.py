#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import torch
import numpy as np
import random
import torch.nn.functional as F
from collections import deque, namedtuple

from model import DQN
import os.path
import time


class LinearSchedule:
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def __call__(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class StatePreproc:
    def __init__(self, device):
        self.device = device
    def __call__(self, x):
        # 网络的默认输入顺序是 batch_size, channel, height, weight
        # input: [B, W, H, C]
        # output: [B, C, H, W]
        x = np.array(x)
        x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = torch.transpose(x, 1, 3)
        x = x / 255.0
        return x


class ReplayBuffer:
    def __init__(self, args):
        self.buffer_size = args.buffer_size // args.num_procs # 整数除法
        self.batch_size = args.batch_size
        self.storage = deque(maxlen=self.buffer_size)
        self.experience = namedtuple('Experience', 
            field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.storage.append(e)
    
    def sample(self):
        exps = random.sample(self.storage, k=self.batch_size)
        # size: [batch_size, ...]
        states = torch.cat([e.state for e in exps if e is not None], dim=0) # size: [batch_size * num_proc, h, w, channel]
        actions = torch.cat([e.action for e in exps if e is not None], dim=0) # size: [batch_szie * num_proc]
        rewards = torch.cat([e.reward for e in exps if e is not None], dim=0) # size: [batch_szie * num_proc]
        next_states = torch.cat([e.next_state for e in exps if e is not None], dim=0)
        dones = torch.cat([e.done for e in exps if e is not None], dim=0) # size: [batch_szie * num_proc]
        batch = (states, actions, rewards, next_states, dones)
        return batch
    
    def __len__(self):
        return len(self.storage)


class Agent(object):
    def __init__(self, args, obs):
        self.net = DQN(args.n_obs, args.n_action)
        self.target_net = DQN(args.n_obs, args.n_action)
        if os.path.isfile('./weights/ckpt.pth'):
            self.net.load_state_dict(torch.load('./weights/ckpt.pth'))
            self.target_net.load_state_dict(torch.load('./weights/ckpt.pth'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_preproc = StatePreproc(self.device)
        self.n_action = args.n_action
        self.gamma = args.gamma
        self.max_grad_norm = args.max_grad_norm
        self.num_procs = args.num_procs
        self.memory = ReplayBuffer(args)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, betas=(0.9,0.99))
        self.criterion = torch.nn.MSELoss()
        # log
        self.log_episode_rewards = torch.zeros(self.num_procs, device=self.device, dtype=torch.float)
        self.episode_rewards = deque([0]*100, maxlen=100)
        self.episode = 1
        self.init(obs)
        # eval
        self.test_episode = args.test_episode
    
    def init(self, obs):
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.obs_tensor = self.state_preproc(obs)# size: [num_proc, 4, height, width]
    
    def act(self, obs, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                q_vals = self.net(obs)
                action = q_vals.argmax(dim=1)
        else:
            action = torch.tensor(np.random.randint(0, self.n_action, size=obs.shape[0]), device=self.device, dtype=torch.int64)
        return action

    def collect_experiences(self, env, num_frames, epsilon):
        for i in range(num_frames):
            actions = self.act(self.obs_tensor, epsilon) # size: [num_proc]
            next_obs, rewards, dones, _ = env.step(actions.cpu().numpy())
            next_obs_tensor = self.state_preproc(next_obs)
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float) # size: [num_proc]
            dones_tensor = 1 - torch.tensor(dones, device=self.device, dtype=torch.float) # size: [num_proc]
            
            self.memory.add(self.obs_tensor, 
                actions, 
                rewards_tensor, 
                next_obs_tensor, 
                dones_tensor)

            self.obs_tensor = next_obs_tensor

            # for log
            self.log_episode_rewards += rewards_tensor
            for i, done in enumerate(dones):
                if done:
                    self.episode_rewards.append(self.log_episode_rewards[i].item())
                    self.log_episode_rewards[i] = 0
                    self.episode += 1
        
        log = {'episode': self.episode, 'average_reward': np.mean(self.episode_rewards)}
        return log
    
    def improve_policy(self, update_times):
        for _ in range(update_times):
            states, acts, rewards, next_states, dones = self.memory.sample()
            with torch.no_grad():
                q_vals = self.target_net(next_states) # next_states size: [batch_size * num_proc, h, w, channel]
                target_max_q = rewards + self.gamma * torch.max(q_vals, 1)[0]
            curr_q_vals = self.net(states)
            curr_max_q = curr_q_vals.gather(1, acts.unsqueeze(1)).squeeze(1)
            # actions = torch.zeros([acts.shape[0], self.n_action], device=self.device, dtype=torch.float)
            # for i, act in enumerate(acts):
            #     actions[i][act.item()] = 1.0
            # curr_max_q = curr_q_vals * actions
            loss = self.criterion(curr_max_q, target_max_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()
        info = {
            'value': curr_max_q.mean().item(),
            'loss': loss.item()
        }
        return info
    
    def update_target_net(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(param.data)
    
    def save_weights(self):
        torch.save(self.net.state_dict(), './weights/ckpt.pth')

    def evaluate(self, env):
        self.net.eval()
        episode_return_list = []
        for i in range(self.test_episode):
            seed = np.random.randint(0, 0xFFFFFF)
            env.seed(seed)
            obs = env.reset()
            done = False
            episode_return = 0
            while not done:
                obs_tensor = self.state_preproc([obs])
                action = self.act(obs_tensor, 0.0)
                obs, reward, done, _ = env.step(action.cpu().numpy())
                episode_return += reward
            episode_return_list.append(episode_return)
        
        info = {'average_return': np.mean(episode_return_list)}
        self.net.train()
        return info
    
    def display(self, env):
        self.net.eval()
        seed = np.random.randint(0, 0xFFFFFF)
        env.seed(seed)
        obs = env.reset()
        need_key = True
        episode = 0
        episode_return = 0
        print('`Enter`: next step\n`E`: Run until end-of-episode\n`Q`: Quit')
        while True:
            if need_key:
                key = input('Press key:')
                if key == 'q':  # quit
                    break
                if key == 'e':  # Run until end-of-episode
                    need_key = False
            env.render()
            obs_tensor = self.state_preproc([obs])
            action = self.act(obs_tensor, 0.0).squeeze(0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode_return += reward
            if not need_key:
                time.sleep(0.1)
            if done:
                episode += 1
                obs = env.reset()
                print('episode: {}, episode_return: {}'.format(episode, episode_return))
                episode_return = 0
                need_key = True
        self.net.train()
