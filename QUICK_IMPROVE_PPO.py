"""
Quick PPO Performance Boost Script

This script integrates the most impactful improvements:
1. PyTorch for proper gradients
2. Enhanced state representation (already in agent_trader.py)
3. Improved reward structure (already in agent_trader.py)
4. Prioritized experience replay
5. Better hyperparameters

Run this instead of agent_trader.py for significantly better performance.

Usage:
    python QUICK_IMPROVE_PPO.py ADBE_daily_data_20260524.csv
"""

import sys
import os

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("✅ PyTorch detected - using improved PPO implementation")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not found - falling back to numpy version")
    print("   Install with: pip install torch")
    print()

# Import the base agent
from agent_trader import (
    run_agent, load_model_predictions, build_lookahead,
    compute_indicators, TradingEnv, compute_metrics,
    backtest, get_current_action, parse_agent_output
)

if TORCH_AVAILABLE:
    from agent_trader_torch import PPOPolicyTorch, train_ppo_torch


def run_improved_agent(csv_file):
    """
    Run agent with all improvements enabled.
    """
    if not TORCH_AVAILABLE:
        print("Falling back to standard agent...")
        return run_agent(csv_file)

    print("="*70)
    print("IMPROVED RL AGENT - PPO with PyTorch")
    print("="*70)

    import pandas as pd
    import numpy as np

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        sys.exit(1)

    df_raw = pd.read_csv(csv_file)
    df_raw = compute_indicators(df_raw).reset_index(drop=True)

    if len(df_raw) < 300:
        print(f"Error: Need at least 300 records, got {len(df_raw)}")
        sys.exit(1)

    ticker = os.path.basename(csv_file).split('_')[0]
    n_records = len(df_raw)
    n_months = n_records / 21.0

    print(f"\nTicker:  {ticker}")
    print(f"Records: {n_records}  (~{n_months:.0f} months)")

    # Layer 1: walk-forward model signals
    print("\n[STEP 1/4] Generating walk-forward model predictions...")
    signals_df = load_model_predictions(csv_file)
    print(f"  Signal rows: {len(signals_df)}")

    if len(signals_df) < 50:
        print("Error: Not enough signal rows to train agent (need >= 50)")
        sys.exit(1)

    # Build lookahead prices
    print("\n[STEP 2/4] Building lookahead price table...")
    lookahead = build_lookahead(signals_df, df_raw)

    # Split signals into train/val
    n_sig = len(signals_df)
    train_cut = int(n_sig * 0.8)
    train_sig = signals_df.iloc[:train_cut].reset_index(drop=True)
    val_sig = signals_df.iloc[train_cut:].reset_index(drop=True)

    # Remap lookahead indices
    train_la = {i: lookahead.get(i, []) for i in range(len(train_sig))}
    val_la = {i: lookahead.get(train_cut + i, []) for i in range(len(val_sig))}

    train_months = len(train_sig) / 21.0
    val_months = len(val_sig) / 21.0

    # Layer 2: Improved PPO training
    print("\n[STEP 3/4] Training improved PPO agent (PyTorch)...")
    np.random.seed(42)
    torch.manual_seed(42)

    # Use improved hyperparameters
    policy = PPOPolicyTorch(
        state_dim=24,      # Enhanced state
        hidden=128,        # Larger network
        action_dim=3,
        lr=1e-4            # Lower learning rate for stability
    )

    train_env = TradingEnv(train_sig, train_la)

    # Check for saved model
    model_path = os.path.join(base_dir, 'ppo_torch_model.pt')
    csv_hash_file = os.path.join(base_dir, 'ppo_torch_csv_hash.txt')
    csv_hash = str(os.path.getsize(csv_file)) + '_' + str(n_records)

    if os.path.exists(model_path) and os.path.exists(csv_hash_file):
        try:
            with open(csv_hash_file) as fh:
                saved_hash = fh.read().strip()
            if saved_hash == csv_hash:
                policy.load_state_dict(torch.load(model_path))
                print("  Warm-started from saved PyTorch model")
            else:
                print("  CSV changed - training from scratch")
        except Exception as e:
            print(f"  Could not load saved model: {e}")

    # Train with more episodes and better hyperparameters
    n_ep = min(max(len(train_sig) * 15, 6000), 30000)  # More episodes
    print(f"  Episodes planned: {n_ep}")

    ep_rewards, outcomes = train_ppo_torch(
        train_env, policy,
        n_episodes=n_ep,
        batch_size=128,     # Larger batches
        gamma=0.99
    )

    print(f"  Training outcomes: TP={outcomes.get('TP',0)} SL={outcomes.get('SL',0)} "
          f"HOLD={outcomes.get('HOLD',0)} TIMEOUT={outcomes.get('TIMEOUT',0)}")

    # Save model
    try:
        torch.save(policy.state_dict(), model_path)
        with open(csv_hash_file, 'w') as fh:
            fh.write(csv_hash)
        print("  Saved PyTorch model")
    except Exception as e:
        print(f"  Warning: could not save model: {e}")

    # Backtest
    print("\n[STEP 4/4] Backtesting on validation set...")
    val_env = TradingEnv(val_sig, val_la)
    trades_df = backtest(val_env, policy)
    metrics = compute_metrics(trades_df, val_months)

    if not metrics:
        print("Warning: No actionable trades in validation set")
        metrics = {'win_rate': 0, 'profit_factor': 0, 'sharpe': 0,
                   'max_drawdown_pct': 0, 'avg_trades_month': 0,
                   'total_trades': 0, 'total_reward': 0,
                   'n_long': 0, 'n_short': 0}

    # Current action - use voting fallback if needed
    use_voting = (metrics['win_rate'] < 40 or metrics['profit_factor'] < 1.0)
    current = get_current_action(signals_df, df_raw, policy, use_voting=use_voting)

    if use_voting:
        print("  Using voting-based fallback (PPO performance below threshold)")

    # Print results
    print("\n" + "="*70)
    print("IMPROVED RL AGENT RESULTS (PyTorch)")
    print("="*70)

    print(f"\nCurrent Decision ({ticker}):")
    print(f"  AGENT_ACTION:     {current['action']}")
    print(f"  AGENT_CONFIDENCE: {current['confidence']:.1f}%")
    print(f"  Entry Price:      ${current['close']:.2f}")
    print(f"  Stop Loss:        ${current['sl']:.2f} ({current['sl_pct']:+.2f}%)")
    print(f"  Take Profit:      ${current['tp']:.2f} ({current['tp_pct']:+.2f}%)")

    print(f"\nValidation Backtest Performance ({len(val_sig)} days, ~{val_months:.0f} months):")
    print(f"  Total Trades:     {metrics['total_trades']}")
    print(f"  AGENT_WINRATE:    {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"  Sharpe Ratio:     {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Avg Trades/Month: {metrics['avg_trades_month']:.1f}")
    print(f"  Long / Short:     {metrics['n_long']} / {metrics['n_short']}")
    print(f"  Total Reward:     {metrics['total_reward']:.2f}")

    # Performance assessment
    print("\nPerformance Assessment:")
    if metrics['win_rate'] >= 40:
        print(f"  ✅ Win Rate:      PASS  ({metrics['win_rate']:.1f}% >= 40%)")
    else:
        print(f"  ⚠️  Win Rate:      BELOW TARGET  ({metrics['win_rate']:.1f}% < 40%)")

    if metrics['profit_factor'] >= 1.5:
        print(f"  ✅ Profit Factor: PASS  ({metrics['profit_factor']:.2f} >= 1.5)")
    else:
        print(f"  ⚠️  Profit Factor: BELOW TARGET  ({metrics['profit_factor']:.2f} < 1.5)")

    if metrics['sharpe'] >= 1.0:
        print(f"  ✅ Sharpe Ratio:  PASS  ({metrics['sharpe']:.2f} >= 1.0)")
    else:
        print(f"  ⚠️  Sharpe Ratio:  BELOW TARGET  ({metrics['sharpe']:.2f} < 1.0)")

    print("\n" + "="*70)

    return current, metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python QUICK_IMPROVE_PPO.py <csv_file>")
        print("Example: python QUICK_IMPROVE_PPO.py ADBE_daily_data_20260524.csv")
        sys.exit(1)

    run_improved_agent(sys.argv[1])
