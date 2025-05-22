import gymnasium as gym
from stable_baselines3 import DQN, PPO 
from custom_reward_env import LLMRewardWrapper
from utils import evaluate_model
import time
import os
from gymnasium.wrappers.record_video import RecordVideo
from llm_interface import generate_prompt

import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

def train_and_evaluate(env, total_timesteps=100000, name="baseline"):
    if isinstance(env, LLMRewardWrapper) and env.strategy == 'dynamic':
        env.refresh_interval = total_timesteps // 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using device:", torch.cuda.get_device_name(0))
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        model = DQN("MlpPolicy", env, verbose=0, device=device, learning_rate=0.0001)
    else:
        model = PPO("MlpPolicy", env, verbose=0, device=device, learning_rate=0.0001)

    start = time.time()
    division = total_timesteps // 10
    # Training loop with step tracking
    for step in range(0, total_timesteps, division):
        model.learn(total_timesteps=division, reset_num_timesteps=False)
        print(f"Steps: {step + division}/{total_timesteps} (Progress Update)") 

    duration = time.time() - start
    try:
        metrics = evaluate_model(model, env)
        print(f"{name} | Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f} | "
              f"Mean Episode Length: {metrics['mean_episode_length']:.2f} | "
              f"Success Rate: {metrics['success_rate']:.2%} | Time: {duration:.2f}s")
    except Exception as e:
        print(f"⚠️ Error during evaluation: {e}")
        metrics = None

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{name.replace(' ', '_')}_model")

    return model, metrics, duration

def visualize_agent(model, env_name="LunarLander-v2", episodes=5):
    """Render episodes live using human-readable display."""
    env = gym.make(env_name, render_mode="human")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.02)
        print(f"Episode {ep + 1} total reward: {total_reward:.2f}")
    env.close()

def record_agent(model, env_name, path="videos", num_episodes=5):
    """Record videos of multiple episodes of the agent."""
    os.makedirs(path, exist_ok=True)
    print(f"Recording {num_episodes} episodes for {env_name}")
    env = gym.make(env_name, render_mode="rgb_array_list")
    env = RecordVideo(env, video_folder=path, episode_trigger=lambda e: e < num_episodes)
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {ep + 1} total reward: {total_reward:.2f}")
    env.close()
    print(f"Videos saved to {path}/")

if __name__ == "__main__":
    # entorno = "LunarLander-v2" # "BipedalWalker-v3"
    entorno = "BipedalWalker-v3"
    generate_prompt(entorno)

    # Baseline
    env1 = gym.make(entorno)
    model1, _, _ = train_and_evaluate(env1, name="Baseline")
    record_agent(model1, env_name=entorno, path=f"videos/baseline/{entorno}")

    # Static LLM reward
    env2 = LLMRewardWrapper(gym.make(entorno), strategy="static")
    model2, _, _ = train_and_evaluate(env2, total_timesteps=100000, name="LLM Static Reward")
    record_agent(model2, env_name=entorno, path=f"videos/static/{entorno}")

    # Dynamic per-step LLM reward
    env3 = LLMRewardWrapper(gym.make(entorno), strategy="dynamic", refresh_interval=100000)
    model3, _, _ = train_and_evaluate(env3, total_timesteps=1000000, name="LLM Dynamic Reward")
    record_agent(model3, env_name=entorno, path=f"videos/dynamic/{entorno}")