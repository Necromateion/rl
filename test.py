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

MINIBATCH_SIZE = 32
REPLAY_REPLAY_MEMORY_SIZE = 1000000
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
DISCOUNT_FACTOR = 0.99
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000
REPLAY_START_SIZE = 50000
NO_OP_MAX = 30

class ConvModel(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=AGENT_HISTORY_LENGTH, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)  
        self.fc2 = nn.Linear(512, num_actions)
        self.opt = optim.Adam(self.parameters(), LEARNING_RATE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_frame(max_frame):
    Luminance_frame = rgb_to_luminance(max_frame).astype(np.uint8)
    resized_frame = cv2.resize(Luminance_frame, dsize=(84, 84))
    return resized_frame

def rgb_to_luminance(frame):
    return 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]

def step_buffer_stack(env, action, buffer):
    im, reward, done, _, _ = env.step(action)
    im = preprocess_frame(im)
    buffer[1:4, :, :] = buffer[0:4-1, :, :]
    buffer[0, :, :] = im
    return buffer.copy(), reward, done

def reset_frame_stack(env, buffer):
    im, _ = env.reset()
    im = preprocess_frame(im)
    buffer = np.stack([im]*4, 0)
    return buffer.copy()

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class ReplayBuffer:
    def __init__(self, buffer_size=REPLAY_REPLAY_MEMORY_SIZE):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.idx = 0

    def insert(self, sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        assert num_samples < min(self.idx, self.buffer_size)
        if self.idx < self.buffer_size:
            return sample(self.buffer[: self.idx], num_samples)
        return sample(self.buffer, num_samples)

def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model, state_transitions, tgt, num_actions, device, gamma=DISCOUNT_FACTOR):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1])for s in state_transitions])).to(device)    
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  

    model.opt.zero_grad()
    qvals = model(cur_states)  
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals * one_hot_actions, -1), rewards.squeeze() + mask[:, 0] * qvals_next * 0.99)

    loss.backward()
    model.opt.step()
    return loss

def main(chkpt=None, device="cuda"):
    do_boltzman_exploration = False

    eps_decay = 0.999999

    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    buffer = np.zeros((4, 84, 84), 'uint8')

    last_observation = reset_frame_stack(env, buffer)

    m = ConvModel(env.action_space.n).to(device)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = ConvModel(env.action_space.n).to(device)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0

    step_num = -1 * REPLAY_START_SIZE

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while True:
            tq.update(1)

            eps = eps_decay ** (step_num)

            if do_boltzman_exploration:
                logits = m(torch.Tensor(last_observation).unsqueeze(0).to(device))[0]
                action = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                if random() < eps:
                    action = env.action_space.sample()  
                else:
                    action = m(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item()

            observation, reward, done = step_buffer_stack(env, action, buffer)
            reward = np.clip(reward, -1, 1)
            rolling_reward += reward

            rb.insert(Sarsd(last_observation, action, reward, observation, done))

            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                rolling_reward = 0
                observation = reset_frame_stack(env, buffer)

            steps_since_train += 1
            step_num += 1

            if (
                rb.idx > REPLAY_START_SIZE
                and steps_since_train > 4000
            ):
                loss = train_step(
                    m, rb.sample(MINIBATCH_SIZE), tgt, env.action_space.n, device
                )
                if epochs_since_tgt % 10 == 0:
                    print(f"""loss: {loss.detach().cpu().item()},eps: {eps},avg_reward: {episode_rewards}""")
                    episode_rewards = []
                epochs_since_tgt += 1

                if epochs_since_tgt > 100:
                    print("updating target model")
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"models/{step_num}.pth")

                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    main()