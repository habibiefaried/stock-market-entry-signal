"""
Curriculum Learning for PPO Agent

Start with easier scenarios where model consensus is clear,
gradually introduce more difficult cases where models disagree.

This helps the agent learn basic strategy before tackling edge cases.
"""

import numpy as np


class CurriculumScheduler:
    """
    Manages curriculum stages during training.

    Stage 1: High model agreement (>5/7 models agree)
    Stage 2: Medium agreement (4/7 models agree)
    Stage 3: Low agreement (models split)
    Stage 4: All scenarios
    """

    def __init__(self, signals_df, n_stages=4):
        self.signals_df = signals_df
        self.n_stages = n_stages
        self.current_stage = 0

        # Classify episodes by difficulty
        self.episodes_by_difficulty = self._classify_episodes()

    def _classify_episodes(self):
        """Classify each episode by model agreement level"""
        MODEL_NAMES = ['xgboost', 'xgboost_heavy', 'lightgbm', 'lightgbm_heavy',
                       'randomforest', 'lstm', 'tft']

        easy = []      # 6-7 models agree
        medium = []    # 4-5 models agree
        hard = []      # models split

        for idx, row in self.signals_df.iterrows():
            signals = [int(row.get(f'{name}_signal', 0)) for name in MODEL_NAMES]

            vote_long = signals.count(1)
            vote_short = signals.count(-1)
            max_agreement = max(vote_long, vote_short)

            if max_agreement >= 6:
                easy.append(idx)
            elif max_agreement >= 4:
                medium.append(idx)
            else:
                hard.append(idx)

        return {
            'easy': np.array(easy),
            'medium': np.array(medium),
            'hard': np.array(hard)
        }

    def get_stage_episodes(self):
        """Return episode indices for current curriculum stage"""
        if self.current_stage == 0:
            # Stage 1: Only easy episodes
            return self.episodes_by_difficulty['easy']

        elif self.current_stage == 1:
            # Stage 2: Easy + medium
            return np.concatenate([
                self.episodes_by_difficulty['easy'],
                self.episodes_by_difficulty['medium']
            ])

        elif self.current_stage == 2:
            # Stage 3: Medium + hard
            return np.concatenate([
                self.episodes_by_difficulty['medium'],
                self.episodes_by_difficulty['hard']
            ])

        else:
            # Stage 4: All episodes
            return np.arange(len(self.signals_df))

    def advance_stage(self):
        """Move to next curriculum stage"""
        if self.current_stage < self.n_stages - 1:
            self.current_stage += 1
            return True
        return False

    def get_progress(self):
        """Return current progress through curriculum"""
        return (self.current_stage + 1) / self.n_stages


def train_with_curriculum(env, policy, curriculum, episodes_per_stage=1500):
    """
    Train PPO agent with curriculum learning.

    Args:
        env: Trading environment
        policy: PPO policy
        curriculum: CurriculumScheduler
        episodes_per_stage: Number of episodes to train at each stage
    """
    print("  Training with curriculum learning...")

    all_rewards = []

    for stage in range(curriculum.n_stages):
        print(f"\n  Curriculum Stage {stage + 1}/{curriculum.n_stages}")

        stage_episodes = curriculum.get_stage_episodes()
        print(f"    Training on {len(stage_episodes)} episodes")

        # Sample from stage episodes
        stage_rewards = []

        for ep in range(episodes_per_stage):
            # Sample an episode from current curriculum stage
            episode_idx = np.random.choice(stage_episodes)

            state = env.reset(idx=episode_idx)
            done = False
            ep_reward = 0

            ep_states = []
            ep_actions = []
            ep_log_probs = []
            ep_rewards = []
            ep_values = []

            while not done:
                action, log_prob, value = policy.act(state)
                next_state, reward, done, info = env.step(action)

                ep_states.append(state)
                ep_actions.append(action)
                ep_log_probs.append(log_prob)
                ep_rewards.append(reward)
                ep_values.append(value)

                ep_reward += reward
                state = next_state

            # Compute returns
            returns = []
            R = 0
            for r in reversed(ep_rewards):
                R = r + 0.99 * R
                returns.insert(0, R)

            advantages = [r - v for r, v in zip(returns, ep_values)]

            # Update policy every 64 episodes
            if (ep + 1) % 64 == 0:
                policy.update(ep_states, ep_actions, ep_log_probs,
                              returns, advantages)

            stage_rewards.append(ep_reward)

            if (ep + 1) % 500 == 0:
                avg = np.mean(stage_rewards[-100:])
                print(f"      Episode {ep+1}/{episodes_per_stage} | avg reward: {avg:.3f}")

        all_rewards.extend(stage_rewards)

        # Advance to next stage if performance is good enough
        avg_stage_reward = np.mean(stage_rewards[-200:])
        if avg_stage_reward > -0.1:  # threshold
            curriculum.advance_stage()
        else:
            print(f"      Stage performance not sufficient ({avg_stage_reward:.3f}), repeating stage...")

    return all_rewards
