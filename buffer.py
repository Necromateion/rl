import random
from typing import Any
import numpy as np
from dataclasses import dataclass

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    new_state: Any
    done: int

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add_experience(self, sarsd: Sarsd):
        if sarsd.new_state.shape != (84, 84, 4):
            raise ValueError(f"Unexpected state shape: {sarsd.new_state.shape}")
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = sarsd
        self.position = (self.position + 1) % self.capacity

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, new_states, dones = zip(
            *[(sarsd.state, sarsd.action, sarsd.reward, sarsd.new_state, sarsd.done) for sarsd in batch]
        )
        return np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(dones)
