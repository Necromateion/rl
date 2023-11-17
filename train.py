import numpy as np
import math
import torch
import torch.nn.functional as F
import logging
import os

from collections import deque
from buffer import *
from var import *
from preprocess import *

# Initialize logging
logging.basicConfig(level=logging.INFO)

NUM_EPISODES = 200
T_MAX = 4000

# usage: new*
def select_action(state, steps_done, policy_net, num_actions, start_eps=INITIAL_EXPLORATION, end_eps=FINAL_EXPLORATION, decay_duration=FINAL_EXPLORATION_FRAME):
    sample = np.random.random()
    eps_threshold = end_eps + (start_eps - end_eps) * math.exp(-1. * steps_done / decay_duration)

    if sample > eps_threshold:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)
            return policy_net(state_tensor).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[np.random.choice(num_actions)]], dtype=torch.long)
def compute_q_values(batch, policy_net, target_net):
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.BoolTensor(done_batch)

    action_batch = action_batch.unsqueeze(-1)

    # Compute Q(s_t, a)
    q_values = policy_net(state_batch).gather(1, action_batch)

    # Compute max_a Q(s_{t+1}, a)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()

    done_batch_float = done_batch.to(torch.float32)

    # Compute target value: r + γ * max_a(Q(s_{t+1}, a))
    target_q_values = reward_batch + (DISCOUNT_FACTOR * next_state_values * (1 - done_batch_float))

    return q_values, target_q_values

def optimize_model(optimizer, q_values, target_q_values):
    loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(dqn_model, target_model, env, replay_buffer):
    optimizer = torch.optim.RMSprop(dqn_model.parameters(), lr=LEARNING_RATE, momentum=GRADIENT_MOMENTUM, eps=MIN_SQUARED_GRADIENT)
    num_actions = env.action_space.n
    steps_done = 0

    episode_rewards = []
    episode_losses = []

    model_save_dir = "saved_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Initialize deque for stacked frames
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for _ in range(AGENT_HISTORY_LENGTH)], maxlen=4)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        state, stacked_frames = process_and_stack_frames(state, stacked_frames, is_new_episode=True)

        episode_reward = 0  # Reset episode reward
        episode_loss = 0    # Reset episode loss

        for t in range(T_MAX):
            steps_done += 1

            # Select and perform an action using epsilon-greedy
            action = select_action(state, steps_done, dqn_model, num_actions)
            next_state, reward, done, _ = env.step(action.item())

            # Stack the frames
            next_state, stacked_frames = process_and_stack_frames(next_state, stacked_frames, is_new_episode=False)

            # Clip the reward
            reward = np.clip(reward, -1, 1)

            episode_reward += reward  # Accumulate episode reward

            # Store the transition in memory
            replay_buffer.add_experience(Sarsd(state, action.item(), reward, next_state, done))

            # Update state
            state = next_state
            # Perform one step of the optimization (on the target network)
            if steps_done % UPDATE_FREQUENCY == 0:
                if len(replay_buffer) > MINIBATCH_SIZE:
                    batch = replay_buffer.sample(MINIBATCH_SIZE)
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
                    
                    state_batch = torch.FloatTensor(state_batch).permute(0, 3, 1, 2)
                    next_state_batch = torch.FloatTensor(next_state_batch).permute(0, 3, 1, 2)
                    
                    modified_batch = (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                    
                    q_values, target_q_values = compute_q_values(modified_batch, dqn_model, target_model)
                    loss = optimize_model(optimizer, q_values, target_q_values)
                    episode_loss += loss
                    
            # Update the target network, copying all weights and biases in DQN
            if steps_done % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                target_model.load_state_dict(dqn_model.state_dict())
                
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss)

        if episode % 10 == 0:
            logging.info(f"Episode: {episode}, Average Reward: {np.mean(episode_rewards[-10:])}, Average Loss: {np.mean(episode_losses[-10:])}")
        if episode % 100 == 0 and episode != 0:
            save_path = os.path.join(model_save_dir, f"dqn_model_episode_{episode}.pth")
            torch.save(dqn_model.state_dict(), save_path)
            logging.info(f"Saved model at episode {episode} to {save_path}")

    return episode_rewards, episode_losses



