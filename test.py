import torch
import numpy as np
from tqdm import tqdm
import gym
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import Any
from random import sample, random
import cv2
from random import randint
import math
from collections import deque

MINIBATCH_SIZE = 32
REPLAY_REPLAY_MEMORY_SIZE = 1000000
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
DISCOUNT_FACTOR = 0.99
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4 * ACTION_REPEAT
FRAME_SKIP = 4
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000
REPLAY_START_SIZE = 50000
NO_OP_MAX = 30
EPS_DECAY_RATE = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_FRAME    
GRADIENT_CLIP = 10

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
 
def preprocess_frame(max_frame):
    Luminance_frame = rgb_to_luminance(max_frame).astype(np.uint8)
    resized_frame = cv2.resize(Luminance_frame, dsize=(84, 84))
    return resized_frame

def rgb_to_luminance(frame):
    return 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]

def select_action(step_num, env, dqn_model, last_observation, device):
    eps = max(FINAL_EXPLORATION, (INITIAL_EXPLORATION - EPS_DECAY_RATE) ** max(1, step_num * step_num))
    if random() < eps:
        action = env.action_space.sample()  
    else:
        action = dqn_model(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item()
    return action

def step_buffer_stack(env, action, buffer):
    im, reward, done, _, _ = env.step(action)
    im = preprocess_frame(im)
    buffer[1:4, :, :] = buffer[0:4-1, :, :]
    buffer[0, :, :] = im
    return buffer, reward, done

def reset_frame_stack(env, buffer):
    im, _ = env.reset()
    im = preprocess_frame(im)
    buffer = np.stack([im]*4, 0)    
    return buffer.copy()

class DuelingDQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=AGENT_HISTORY_LENGTH, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc_value = nn.Linear(7 * 7 * 64, 512)
        self.fc_advantage = nn.Linear(7 * 7 * 64, 512)
        self.value_output = nn.Linear(512, 1)
        self.advantage_output = nn.Linear(512, num_actions)
        self.opt = optim.Adam(self.parameters(), LEARNING_RATE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        value = F.relu(self.fc_value(x))
        advantage = F.relu(self.fc_advantage(x))
        value = self.value_output(value)
        advantage = self.advantage_output(advantage)
        return value + advantage - advantage.mean(1, keepdim=True)

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size=REPLAY_REPLAY_MEMORY_SIZE):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = 0.6  

    def insert(self, sars, error):
        priority = (error + 1e-5) ** self.alpha
        self.buffer.append(sars)
        self.priorities.append(priority)        

    def sample(self, num_samples):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probs = scaled_priorities / sum(scaled_priorities)
        indices = np.random.choice(len(self.buffer), num_samples, p=sample_probs)
        samples = [self.buffer[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (error + 1e-5) ** self.alpha
            self.priorities[idx] = priority

def train_step(model, state_transitions, target_model, num_actions, device, replay_buffer, indices, gamma=DISCOUNT_FACTOR):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)    
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        next_state_actions = model(next_states).max(-1)[1]
        qvals_next = target_model(next_states).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)

    model.opt.zero_grad()
    qvals = model(cur_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss_fn = nn.SmoothL1Loss()
    target_values = rewards.squeeze() + mask[:, 0] * qvals_next * gamma
    predicted_values = torch.sum(qvals * one_hot_actions, -1)
    loss = loss_fn(predicted_values, target_values)
    loss.backward()
    model.opt.step()

    errors = torch.abs(predicted_values - target_values).data.cpu().numpy()
    replay_buffer.update_priorities(indices, errors)
    return loss, errors

def main(device="cuda"):
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    buffer = np.zeros((4, 84, 84), 'uint8')
    last_observation = reset_frame_stack(env, buffer)

    dqn_model = DuelingDQN(env.action_space.n).to(device)
    target_model = DuelingDQN(env.action_space.n).to(device)
    target_model.load_state_dict(dqn_model.state_dict())

    replay_buffer = PrioritizedReplayBuffer()
    steps_before_update = 0
    step_num = -REPLAY_START_SIZE
    episode_rewards = []
    episode_loss = []
    tmp_reward = 0
    tq = tqdm()
    max_priority = 1e-6
    for _ in range(100000000):
        tq.update(1)

        if step_num % ACTION_REPEAT == 0:
            action = select_action(step_num, env, dqn_model, last_observation, device)
        observation, reward, done = step_buffer_stack(env, action, buffer)
        reward = np.clip(reward, -1, 1)
        tmp_reward += reward

        error = max_priority
        replay_buffer.insert(Sarsd(last_observation, action, reward, observation, done), error)
        last_observation = observation

        if done:
            episode_rewards.append(tmp_reward)
            tmp_reward = 0
            observation = reset_frame_stack(env, buffer)

        steps_before_update += 1
        step_num += 1

        if (step_num > 0 and step_num % UPDATE_FREQUENCY == 0):
            transitions, indices = replay_buffer.sample(MINIBATCH_SIZE)
            loss, new_priorities = train_step(dqn_model, transitions, target_model, env.action_space.n, device, replay_buffer, indices)
            episode_loss.append(loss.detach().cpu().item())
            max_priority = max(max_priority, max(new_priorities))

        if (step_num > 0 and steps_before_update > TARGET_NETWORK_UPDATE_FREQUENCY):
            print(f"""step : {step_num}, eps : {0.999999 ** step_num}, loss_average: {np.mean(episode_loss)},avg_reward: {np.mean(episode_rewards)}, max_reward: {np.max(episode_rewards)}""")  
            target_model.load_state_dict(dqn_model.state_dict())      
            if(np.mean(episode_rewards)) > 2:           
                torch.save(target_model.state_dict(), f"models/{step_num}.pth")
            episode_rewards = []      
            episode_loss = []  
            steps_before_update = 0

    env.close()

if __name__ == "__main__":
    main()
