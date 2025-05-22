import numpy as np

def evaluate_model(model, env, episodes=20):
    """
    Evaluate a model on an environment and compute detailed metrics.
    
    Args:
        model: Trained RL model (e.g., from Stable Baselines3).
        env: Gymnasium environment.
        episodes: Number of episodes to evaluate (default: 20).
    
    Returns:
        dict: Metrics including mean reward, std reward, mean episode length,
              success rate, min reward, and max reward.
    """
    rewards = []
    episode_lengths = []
    successes = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Check for success in info (environment-specific)
            if 'success' in info:
                successes.append(1 if info['success'] else 0)
            
        rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Compute metrics
    metrics = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'success_rate': np.mean(successes) if successes else 0.0
    }
    
    return metrics