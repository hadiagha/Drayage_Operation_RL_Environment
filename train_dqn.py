#!/usr/bin/env python3
"""
DQN Training Script for Drayage Trailer Dispatching Environment

This script trains a Deep Q-Network (DQN) agent to learn optimal
trailer repositioning policies using Stable-Baselines3.

Usage:
    python train_dqn.py                    # Train with default settings
    python train_dqn.py --timesteps 500000 # Train for 500k steps
    python train_dqn.py --eval             # Evaluate a trained model
"""

import argparse
import os
import numpy as np
from datetime import datetime

from drayage_trailer_dispatching_env import DrayageEnv


def create_environment(seed: int = 42, for_eval: bool = False):
    """
    Create the Drayage environment with a realistic configuration.
    
    Args:
        seed: Random seed for reproducibility
        for_eval: If True, use terminate_on_forbidden=True for strict evaluation
    """
    num_trailers = 6
    num_yards = 4
    
    # Diverse orders throughout the day
    orders_config = [
        # Morning pickups
        {'pickup_tw': (7.0, 9.0), 'delivery_tw': (11.0, 13.0), 'pickup_duration': 30, 'delivery_duration': 25, 'cost': 100.0},
        {'pickup_tw': (7.5, 9.5), 'delivery_tw': (12.0, 14.0), 'pickup_duration': 35, 'delivery_duration': 30, 'cost': 100.0},
        {'pickup_tw': (8.0, 10.0), 'delivery_tw': (13.0, 15.0), 'pickup_duration': 40, 'delivery_duration': 25, 'cost': 100.0},
        # Mid-morning pickups
        {'pickup_tw': (9.0, 11.0), 'delivery_tw': (14.0, 16.0), 'pickup_duration': 30, 'delivery_duration': 20, 'cost': 100.0},
        {'pickup_tw': (9.5, 11.5), 'delivery_tw': (15.0, 17.0), 'pickup_duration': 45, 'delivery_duration': 35, 'cost': 100.0},
        # Late morning pickups
        {'pickup_tw': (10.0, 12.0), 'delivery_tw': (15.0, 17.0), 'pickup_duration': 35, 'delivery_duration': 30, 'cost': 100.0},
        {'pickup_tw': (10.5, 12.5), 'delivery_tw': (16.0, 18.0), 'pickup_duration': 30, 'delivery_duration': 25, 'cost': 100.0},
        # Afternoon pickups
        {'pickup_tw': (11.0, 13.0), 'delivery_tw': (17.0, 19.0), 'pickup_duration': 40, 'delivery_duration': 30, 'cost': 100.0},
        {'pickup_tw': (12.0, 14.0), 'delivery_tw': (18.0, 20.0), 'pickup_duration': 35, 'delivery_duration': 25, 'cost': 100.0},
        {'pickup_tw': (13.0, 15.0), 'delivery_tw': (19.0, 21.0), 'pickup_duration': 30, 'delivery_duration': 20, 'cost': 100.0},
    ]
    
    # Travel times between yards (asymmetric for realism)
    travel_times = np.array([
        [0, 45, 60, 35],
        [50, 0, 30, 55],
        [65, 35, 0, 45],
        [40, 60, 50, 0]
    ])
    
    # Yard configurations
    yard_min_empty = [2, 1, 1, 2]
    yard_importance = [1.0, 0.8, 0.7, 0.9]
    yard_attach_times = [10, 12, 8, 10]
    
    env = DrayageEnv(
        num_trailers=num_trailers,
        num_yards=num_yards,
        orders_config=orders_config,
        travel_time_matrix=travel_times,
        yard_min_empty=yard_min_empty,
        yard_importance=yard_importance,
        yard_attach_times=yard_attach_times,
        step_minutes=15,
        min_hour=6,
        max_hour=22,
        n_step_deficit=4,
        delay_cost_per_step=70.0,
        forbidden_action_penalty=30.0,
        laplace_scale=1.0,
        terminate_on_forbidden=for_eval,  # Strict for eval, lenient for training
        seed=seed
    )
    
    return env


def train(
    total_timesteps: int = 100000,
    learning_rate: float = 1e-4,
    buffer_size: int = 50000,
    batch_size: int = 64,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    gamma: float = 0.99,
    train_freq: int = 4,
    target_update_interval: int = 1000,
    seed: int = 42,
    save_path: str = "models",
    log_path: str = "logs"
):
    """
    Train a DQN agent on the Drayage environment.
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate for the optimizer
        buffer_size: Size of the replay buffer
        batch_size: Minibatch size for training
        exploration_fraction: Fraction of training for epsilon decay
        exploration_final_eps: Final exploration rate
        gamma: Discount factor
        train_freq: Update the model every train_freq steps
        target_update_interval: Update target network every N steps
        seed: Random seed
        save_path: Directory to save models
        log_path: Directory for tensorboard logs
    """
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("ERROR: stable-baselines3 is required for training.")
        print("Install with: pip install stable-baselines3[extra]")
        return None
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dqn_drayage_{timestamp}"
    
    print("=" * 60)
    print("DQN Training for Drayage Environment")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Buffer size: {buffer_size:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gamma: {gamma}")
    print(f"  Exploration fraction: {exploration_fraction}")
    print(f"  Final epsilon: {exploration_final_eps}")
    print(f"  Seed: {seed}")
    print()
    
    # Create training environment
    print("Creating training environment...")
    train_env = create_environment(seed=seed, for_eval=False)
    train_env = Monitor(train_env)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_environment(seed=seed + 1000, for_eval=True)
    eval_env = Monitor(eval_env)
    
    # Create DQN model
    print("Initializing DQN model...")
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        tau=1.0,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log=log_path,
        seed=seed,
        policy_kwargs=dict(
            net_arch=[128, 128, 64]  # 3-layer network
        )
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, run_name),
        log_path=os.path.join(log_path, run_name),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_path, run_name, "checkpoints"),
        name_prefix="dqn_checkpoint"
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=run_name,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, run_name, "final_model")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nTo view training logs, run:")
    print(f"  tensorboard --logdir {log_path}")
    print(f"\nTo evaluate the model, run:")
    print(f"  python train_dqn.py --eval --model {final_model_path}")
    
    return model


def evaluate(model_path: str, num_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained DQN model.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    """
    try:
        from stable_baselines3 import DQN
    except ImportError:
        print("ERROR: stable-baselines3 is required.")
        print("Install with: pip install stable-baselines3[extra]")
        return
    
    print("=" * 60)
    print("Evaluating DQN Model")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Episodes: {num_episodes}")
    print()
    
    # Load model
    model = DQN.load(model_path)
    
    # Create evaluation environment
    env = create_environment(seed=12345, for_eval=True)
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    orders_delivered = []
    total_delays = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render(mode='human')
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        orders_delivered.append(info['orders_delivered'])
        total_delays.append(info['total_delay_cost'])
        
        print(f"Episode {ep + 1}: Reward={total_reward:.2f}, "
              f"Steps={steps}, Delivered={info['orders_delivered']}, "
              f"Delay=${info['total_delay_cost']:.0f}")
    
    env.close()
    
    # Print summary
    print("\n" + "-" * 60)
    print("Evaluation Summary")
    print("-" * 60)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Mean Orders Delivered: {np.mean(orders_delivered):.1f}")
    print(f"Mean Delay Cost: ${np.mean(total_delays):.0f}")


def compare_with_baseline(model_path: str = None, num_episodes: int = 10):
    """
    Compare DQN agent with random and greedy baselines.
    """
    from drayage_renderer import smart_policy
    
    try:
        from stable_baselines3 import DQN
        has_model = model_path is not None
        if has_model:
            model = DQN.load(model_path)
    except ImportError:
        has_model = False
    
    print("=" * 60)
    print("Baseline Comparison")
    print("=" * 60)
    
    results = {}
    
    # Test each policy
    policies = ['random', 'greedy']
    if has_model:
        policies.append('dqn')
    
    for policy_name in policies:
        env = create_environment(seed=12345, for_eval=False)
        
        rewards = []
        delivered = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                if policy_name == 'random':
                    action = env.action_space.sample()
                elif policy_name == 'greedy':
                    action = smart_policy(env, obs)
                else:  # dqn
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            rewards.append(total_reward)
            delivered.append(info['orders_delivered'])
        
        env.close()
        
        results[policy_name] = {
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'delivered_mean': np.mean(delivered)
        }
        
        print(f"\n{policy_name.upper()} Policy:")
        print(f"  Mean Reward: {results[policy_name]['reward_mean']:.2f} ± {results[policy_name]['reward_std']:.2f}")
        print(f"  Mean Delivered: {results[policy_name]['delivered_mean']:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train DQN on Drayage Environment")
    
    # Mode
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained model")
    parser.add_argument("--compare", action="store_true", help="Compare with baselines")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Paths
    parser.add_argument("--model", type=str, default=None, help="Path to model for evaluation")
    parser.add_argument("--save-path", type=str, default="models", help="Model save directory")
    parser.add_argument("--log-path", type=str, default="logs", help="Tensorboard log directory")
    
    # Evaluation
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    args = parser.parse_args()
    
    if args.eval:
        if args.model is None:
            print("ERROR: --model path required for evaluation")
            return
        evaluate(args.model, args.episodes, args.render)
    elif args.compare:
        compare_with_baseline(args.model, args.episodes)
    else:
        train(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            seed=args.seed,
            save_path=args.save_path,
            log_path=args.log_path
        )


if __name__ == "__main__":
    main()
