"""
Improved PPO RL Agent using PyTorch
Much better gradient computation and training stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOPolicyTorch(nn.Module):
    """
    Improved PPO policy with proper backpropagation via PyTorch.
    Much more stable and effective than the numpy version.
    """

    def __init__(self, state_dim=16, hidden=128, action_dim=3, lr=3e-4):
        super().__init__()

        # Policy network (actor) - deeper architecture
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, action_dim)
        )

        # Value network (critic) - separate from actor
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    def forward(self, state):
        """Returns action probabilities and state value"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        value = self.critic(state)
        return probs, value

    def act(self, state):
        """Sample action from policy"""
        with torch.no_grad():
            probs, value = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def act_greedy(self, state):
        """Greedy action selection (for inference)"""
        with torch.no_grad():
            probs, value = self.forward(state)
            action = torch.argmax(probs)
        return action.item(), probs[action].item(), value.item()

    def update(self, states, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, n_epochs=10):
        """
        Proper PPO update with gradient descent.

        Key improvements over numpy version:
        - Actual backpropagation through computation graph
        - Multiple epochs over batch for better sample efficiency
        - Separate actor-critic networks
        - Layer normalization for stability
        - Proper advantage normalization
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(n_epochs):
            # Forward pass
            probs, values = self.forward(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


def train_ppo_torch(env, policy, n_episodes=5000, batch_size=64, gamma=0.99):
    """
    Improved training loop with better exploration and stability.
    """
    print(f"  Training PPO agent with PyTorch ({n_episodes} episodes)...")

    ep_rewards = []
    outcomes = {'TP': 0, 'SL': 0, 'HOLD': 0, 'TIMEOUT': 0, 'NEUTRAL': 0}

    # Experience buffer
    buffer_states = []
    buffer_actions = []
    buffer_log_probs = []
    buffer_rewards = []
    buffer_values = []
    buffer_dones = []

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_rew = 0.0

        ep_states = []
        ep_actions = []
        ep_log_probs = []
        ep_rewards = []
        ep_values = []

        # Collect trajectory
        while not done:
            action, log_prob, value = policy.act(state)
            next_state, reward, done, info = env.step(action)

            ep_states.append(state.copy())
            ep_actions.append(action)
            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
            ep_values.append(value)

            ep_rew += reward
            state = next_state

        # Compute returns and advantages (GAE)
        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(ep_rewards))):
            delta = ep_rewards[t] + gamma * next_value - ep_values[t]
            gae = delta + gamma * 0.95 * gae  # GAE(λ=0.95)
            returns.insert(0, gae + ep_values[t])
            advantages.insert(0, gae)
            next_value = ep_values[t]

        # Add to buffer
        buffer_states.extend(ep_states)
        buffer_actions.extend(ep_actions)
        buffer_log_probs.extend(ep_log_probs)
        buffer_rewards.extend(returns)
        buffer_values.extend(advantages)

        outcome = info.get('outcome', 'NEUTRAL')
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        ep_rewards.append(ep_rew)

        # Update policy when buffer is full
        if len(buffer_states) >= batch_size or ep == n_episodes - 1:
            metrics = policy.update(
                buffer_states, buffer_actions, buffer_log_probs,
                buffer_rewards, buffer_values,
                n_epochs=10
            )

            buffer_states = []
            buffer_actions = []
            buffer_log_probs = []
            buffer_rewards = []
            buffer_values = []

        # Logging
        if (ep + 1) % 500 == 0:
            avg = np.mean(ep_rewards[-100:]) if len(ep_rewards) >= 100 else np.mean(ep_rewards)
            print(f"    Episode {ep+1}/{n_episodes} | avg reward: {avg:.3f}")

    return ep_rewards, outcomes
