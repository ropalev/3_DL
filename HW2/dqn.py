import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

import math
import random
import numpy as np

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def store(self, exptuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exptuple
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self, n_cols, n_rows):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.conv_out_size = 6
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(1, self.conv_out_size, kernel_size=3, stride=1, padding=1)
        self.l2 = nn.Linear(self.conv_out_size * n_cols * n_rows, n_cols * n_rows)


    def forward(self, state):
        x = F.relu(self.conv(state.view(-1, 1, self.n_cols, self.n_rows)))
        x = x.view(-1, self.conv_out_size * self.n_cols * self.n_rows)
        x = self.l2(x)
        return x


class DQN():
    def __init__(self, env):
        self.name = 'DQN'
        self.env = env
        self.model = Network(self.env.n_cols, self.env.n_rows)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), 0.001) #00.1
        self.steps_done = 0
        self.episode_durations = []
        self.mean_rewards = []
        self.winrates = []
        self.gamma = 0.9
        self.batch_size = 64
        self.eps_init, self.eps_final, self.eps_decay = 0.9, 0.01, 200
        self.num_step = 0


    def action_against_random_agent(self, action):
          (next_state, empty_spaces, cur_turn), reward, done, info = self.env.step(action)
          if done:
              return (next_state, empty_spaces, cur_turn), reward, done, info
          return self.env.step(empty_spaces[np.random.randint(len(empty_spaces))])


    def explotation_action_(self, state, emptySpaces):
        state = state.flatten()
        sorted_actions = self.model(state).data.argsort().numpy().reshape(-1,1)
        for actions in reversed(sorted_actions):
            if actions not in emptySpaces:
                continue
            return self.env.action_from_int(actions)


    def explotation_action(self, state, emptySpaces, env=None):
        with torch.no_grad():
          state = env.board.flatten()
          state_tensor = torch.tensor([state], dtype=torch.float32)
          emptySpaces = env.emptySpaces
          if emptySpaces is None:
            emptySpaces = [[i, j] for i in range(env.n_rows) for j in range(env.n_cols)]
          if len(emptySpaces) == 0:
                  return -1
          emptySpaces = [env.n_cols * i[0] + i[1] for i in emptySpaces]
          state = state_tensor.flatten()
          sorted_actions = self.model(state).data.argsort().numpy().reshape(-1,1)
          for actions in reversed(sorted_actions):
              if actions not in emptySpaces:
                  continue
              return env.action_from_int(actions)


    def get_action(self, state):
        emptySpaces = self.env.emptySpaces
        if emptySpaces is None:
            emptySpaces = [[i, j] for i in range(self.env.n_rows) for j in range(self.env.n_cols)]
        if len(emptySpaces) == 0:
                return -1
        emptySpaces = [self.env.n_cols * i[0] + i[1] for i in emptySpaces]
        eps_threshold = self.eps_final + (self.eps_init - self.eps_final) * math.exp(-1. * self.num_step / self.eps_decay)
        if np.random.rand() > eps_threshold:
            action = self.explotation_action_(state, emptySpaces)
            action = action[0] * self.env.n_rows + action[1]
        else:
            action = np.random.choice(emptySpaces)
        return torch.tensor([[action]], dtype=torch.int64)


    def run_episode(self, e=0, do_learning=True, greedy=False, render=False):
        self.env.reset()
        state, num_step = self.env.board.flatten(), 0
        empty_spaces = [[i, j] for i in range(self.env.n_rows) for j in range(self.env.n_cols)]
        while True:
            state_tensor = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                if greedy:
                    action = self.explotation_action_(state_tensor, empty_spaces)
                else:
                    action = self.get_action(state_tensor)
            action_tuple = self.env.action_from_int(action.numpy()[0][0])
            _, reward, done, _ = self.action_against_random_agent(action_tuple)
            next_state = self.env.board.flatten()
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
            transition = (state_tensor, action, next_state_tensor, torch.tensor([reward], dtype=torch.float32))
            self.memory.store(transition)
            if do_learning:
                self.learn()
            state = next_state
            num_step += 1
            if done:
                self.episode_durations.append(num_step)
                break


    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        batch_reward = Variable(torch.cat(batch_reward))
        batch_next_state = Variable(torch.cat(batch_next_state))
        Q = self.model(batch_state).gather(1, batch_action).reshape([self.batch_size])
        Qmax = self.model(batch_next_state).detach().max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)
        loss = F.smooth_l1_loss(Q, Qnext)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def play_against_random_agent(self):
        self.model.eval()
        self.env.reset()
        state, num_step = self.env.board.flatten(), 0
        emptySpaces = [[i, j] for i in range(self.env.n_rows) for j in range(self.env.n_cols)]
        emptySpaces = [self.env.n_cols * i[0] + i[1] for i in emptySpaces]
        while True:
            state_tensor = torch.tensor([state], dtype=torch.float32)
            action_tuple = self.explotation_action_(state_tensor, emptySpaces)
            _, reward, done, _ = self.action_against_random_agent(action_tuple)
            next_state = self.env.board.flatten()
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
            state = next_state
            num_step += 1
            emptySpaces = self.env.emptySpaces
            emptySpaces = [self.env.n_cols * i[0] + i[1] for i in emptySpaces]
            if done:
                break
        return reward, done


    def validate_against_random_agent_with_winrate(self, num_episodes=1000):
        wins = 0
        for i in range(num_episodes):
            reward, done = self.play_against_random_agent()
            if done:
              if reward == 1:
                wins += 1
        self.winrates.append(wins/num_episodes)
