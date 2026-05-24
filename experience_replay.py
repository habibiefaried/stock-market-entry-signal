"""
Experience Replay Buffer for PPO

Improves sample efficiency by allowing the agent to learn from past experiences
multiple times. Particularly useful when training data is limited.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Stores past experiences and allows sampling for training.
    """

    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done, log_prob, value):
        """Add an experience to the buffer"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            'states': np.array([exp['state'] for exp in batch]),
            'actions': [exp['action'] for exp in batch],
            'rewards': [exp['reward'] for exp in batch],
            'next_states': np.array([exp['next_state'] for exp in batch]),
            'dones': [exp['done'] for exp in batch],
            'log_probs': [exp['log_prob'] for exp in batch],
            'values': [exp['value'] for exp in batch]
        }

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay - samples important experiences more frequently.

    Prioritizes:
    - Experiences with high TD error (where the agent was most wrong)
    - Rare outcomes (TP hits are less common than timeouts)
    - Recent experiences (more relevant to current policy)
    """

    def __init__(self, max_size=10000, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.max_size = max_size
        self.alpha = alpha  # how much prioritization to use (0 = uniform, 1 = full priority)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done, log_prob, value, outcome='NEUTRAL'):
        """Add experience with priority"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value,
            'outcome': outcome
        }

        # Initial priority: higher for rare outcomes
        if outcome == 'TP':
            priority = self.max_priority * 2.0  # TP is rare and valuable
        elif outcome == 'SL':
            priority = self.max_priority * 1.5  # SL is also important to learn from
        else:
            priority = self.max_priority

        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # Replace lowest priority item
            min_idx = np.argmin(self.priorities)
            self.buffer[min_idx] = experience
            self.priorities[min_idx] = priority

    def sample(self, batch_size, beta=0.4):
        """
        Sample batch with prioritization

        beta: importance sampling weight (0 = no correction, 1 = full correction)
        """
        if len(self.buffer) == 0:
            return None

        # Calculate sampling probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)),
                                   p=probs, replace=False)

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize

        batch = [self.buffer[idx] for idx in indices]

        return {
            'states': np.array([exp['state'] for exp in batch]),
            'actions': [exp['action'] for exp in batch],
            'rewards': [exp['reward'] for exp in batch],
            'next_states': np.array([exp['next_state'] for exp in batch]),
            'dones': [exp['done'] for exp in batch],
            'log_probs': [exp['log_prob'] for exp in batch],
            'values': [exp['value'] for exp in batch],
            'weights': weights,
            'indices': indices
        }

    def update_priorities(self, indices, priorities):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)
