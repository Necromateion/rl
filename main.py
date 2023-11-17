import logging
from collections import deque
import numpy as np
import gym


from preprocess import *
from var import *
from DNN import *
from buffer import *
from train import *

def main():  
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array') 
    initial_state, _ = env.reset()

    stacked_frames = deque([np.zeros((84, 84), dtype=int) for _ in range(AGENT_HISTORY_LENGTH)], maxlen=AGENT_HISTORY_LENGTH)

    initial_state, stacked_frames = process_and_stack_frames(initial_state, stacked_frames, is_new_episode=True)

    num_actions = env.action_space.n
    dqn_model = Deep_neural_network(num_actions)
    target_model = Deep_neural_network(num_actions)
    target_model.load_state_dict(dqn_model.state_dict())  

    replay_buffer = ReplayBuffer(capacity=REPLAY_MEMORY_SIZE)

    state = initial_state

    for _ in range(REPLAY_START_SIZE):
        action = np.random.choice(num_actions)  # Random initial actions
        next_state, reward, done, _, _ = env.step(action)

        # Stack the frames
        next_stacked_frames = process_and_stack_frames(next_state, stacked_frames, is_new_episode=False)

        replay_buffer.add_experience(sarsd=(state, action, reward, next_state, done))

        if done:
            state = env.reset()
            state, stacked_frames = process_and_stack_frames(state, stacked_frames, is_new_episode=True)
        else:
            state = next_state


    train(dqn_model, target_model, env, replay_buffer)


if __name__ == "__main__":
    main()
