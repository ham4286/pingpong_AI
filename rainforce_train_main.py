import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pygame
from collections import deque

#신경망 모델
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#DQN 에이전트 설정
class DQNAgent:
    def __init__(self, input_dim, output_dim, lr = 0.001, gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.01):
        self.memory = deque(maxlen = 2000)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.output_dim = output_dim

        self.q_network = DQN(input_dim, output_dim)
        self.target_network = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = lr)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_transition(self, state, action, reward,ai_reward, player_reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done, ai_reward, player_reward))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.output_dim))

        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self, batch_size = 64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions).squeeze()
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


#핑퐁 학습
class PongEnv:
    def __init__(self):
        self.width, self.height = 480 , 640
        self.paddle_width, self.paddle_height = 160, 20
        self.ball_radius = 16
        self.paddle_y = self.paddle_height
        self.reset()
        
    def reset(self):
        self.paddle_x = (self.width - self.paddle_width) / 2
        self.player_paddle_x = self.width / 2 - self.paddle_width / 2 
        self.ball_x, self.ball_y = self.width / 2, self.height / 2
        self.ball_dx, self.ball_dy = np.random.choice([-3, 3]), -abs(np.random.choice([-3, 3]))
        self.ball_dy = abs(self.ball_dy)
        self.done = False

        return (self.paddle_x, self.ball_x, self.ball_y)

        

    def step(self, action):
        ai_reward = 0
        player_reward = 0
        reward = 0
        if action == 0:
            self.paddle_x -= 5
        
        elif action == 2:
            self.paddle_x += 5
        
        self.paddle_x = np.clip(self.paddle_x, 0, self.width - self.paddle_width)

        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        if self.ball_x <= 0 or self.ball_x >= self.width:
            self.ball_dx *= -1
        
        if self.ball_y >= self.height:
            self.ball_dy *= -1

        if self.ball_y <= self.paddle_y + self.paddle_height  and self.paddle_x < self.ball_x < self.paddle_x + self.paddle_width:
            self.ball_dy *= -1
            ai_reward = 1
        
        if self.ball_y >= self.height - self.paddle_height and self.player_paddle_x < self.ball_x < self.player_paddle_x + self.paddle_width:
            self.ball_dy*= -1
            player_reward = 1



        elif self.ball_y <= 0:
            self.done = True
            ai_reward = -1
        
        elif self.ball_y >= self.height:
            self.done = True
            player_reward = -1
        else:
            reward = 0
            

        
        return (self.paddle_x, self.ball_x, self.ball_y), reward, self.done, ai_reward, player_reward



