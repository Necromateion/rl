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

class ConvModel(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)  
        self.fc2 = nn.Linear(512, num_actions)
        self.opt = optim.Adam(self.parameters(), lr=0.00025)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def setup_frame_stacking_and_resizing(env, w, h, num_stack=4):
    buffer = np.zeros((num_stack, h, w), 'uint8')
    frame = None
    return env, buffer, frame, w, h, num_stack

def preprocess_frame(max_frame, w, h):
    Luminance_frame = rgb_to_luminance(max_frame).astype(np.uint8)
    resized_frame = cv2.resize(Luminance_frame, dsize=(84, 84))
    return resized_frame

def rgb_to_luminance(frame):
    return 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]

def step_frame_stack(env, action, buffer, w, h, num_stack):
    im, reward, done, info, _ = env.step(action)
    frame = im.copy()
    im = preprocess_frame(im, w, h)
    buffer[1:num_stack, :, :] = buffer[0:num_stack-1, :, :]
    buffer[0, :, :] = im
    return buffer.copy(), reward, done, info, frame

def render_frame_stack(env, mode, frame):
    if mode == 'rgb_array':
        return frame
    return env.render(mode)

def reset_frame_stack(env, buffer, w, h, num_stack):
    im, _ = env.reset()
    frame = im.copy()
    im = preprocess_frame(im, w, h)
    buffer = np.stack([im]*num_stack, 0)
    return buffer.copy(), frame

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    def __init__(self, buffer_size=100000):
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

def train_step(model, state_transitions, tgt, num_actions, device, gamma=0.99):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(
        device
    )
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(
        device
    )
    mask = torch.stack(
        (
            [
                torch.Tensor([0]) if s.done else torch.Tensor([1])
                for s in state_transitions
            ]
        )
    ).to(device)
    next_states = torch.stack(
        ([torch.Tensor(s.next_state) for s in state_transitions])
    ).to(device)
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  # (N, num_actions)

    model.opt.zero_grad()
    qvals = model(cur_states)  # (N, num_actions)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(
        torch.sum(qvals * one_hot_actions, -1), rewards.squeeze() + mask[:, 0] * qvals_next * 0.99
    )

    loss.backward()
    model.opt.step()
    return loss

def run_test_episode(model, env, device, max_steps=1000):  # -> reward, movie?
    frames = []
    obs = env.reset()
    frames.append(env.frame)

    idx = 0
    done = False
    reward = 0
    while not done and idx < max_steps:
        action = model(torch.Tensor(obs).unsqueeze(0).to(device)).max(-1)[-1].item()
        obs, r, done, _, _ = env.step(action)
        reward += r
        frames.append(env.frame)
        idx += 1

    return reward, np.stack(frames, 0)

def main(test=False, chkpt=None, device="cuda"):
    do_boltzman_exploration = False
    memory_size = 1000000
    min_rb_size = 50000
    sample_size = 32
    lr = 0.0001

    # eps_max = 1.0
    eps_min = 0.1

    eps_decay = 0.999999

    env_steps_before_train = 16
    tgt_model_update = 5000
    epochs_before_test = 1500

    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    env, buffer, frame, w, h, num_stack = setup_frame_stacking_and_resizing(env, 84, 84, 4)

    test_env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    test_env, test_buffer, test_frame, _, _, _ = setup_frame_stacking_and_resizing(test_env, 84, 84, 4)

    last_observation, last_frame = reset_frame_stack(env, buffer, w, h, num_stack)

    m = ConvModel(env.action_space.n).to(device)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = ConvModel(env.action_space.n).to(device)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0
    epochs_since_test = 0

    step_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while True:
            if test:
                render_frame_stack(env, 'human', last_frame)
                time.sleep(0.05)
            tq.update(1)

            eps = eps_decay ** (step_num)
            if test:
                eps = 0

            if do_boltzman_exploration:
                logits = m(torch.Tensor(last_observation).unsqueeze(0).to(device))[0]
                action = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                if random() < eps:
                    action = env.action_space.sample()  # Random action
                else:
                    action = m(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item()

            observation, reward, done, info, frame = step_frame_stack(env, action, buffer, w, h, num_stack)
            rolling_reward += reward

            rb.insert(Sarsd(last_observation, action, reward, observation, done))

            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0
                observation, frame = reset_frame_stack(env, buffer, w, h, num_stack)

            steps_since_train += 1
            step_num += 1

            if (
                (not test)
                and rb.idx > min_rb_size
                and steps_since_train > env_steps_before_train
            ):
                loss = train_step(
                    m, rb.sample(sample_size), tgt, env.action_space.n, device
                )
                if step_num % 10 == 0:
                    print(f"""loss: {loss.detach().cpu().item()},eps: {eps},avg_reward: {np.mean(episode_rewards)}""")
                episode_rewards = []
                epochs_since_tgt += 1
                epochs_since_test += 1

                if epochs_since_test > epochs_before_test:
                    rew, frames = run_test_episode(m, test_env, device)
                    epochs_since_test = 0

                if epochs_since_tgt > tgt_model_update:
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