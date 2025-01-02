import gymnasium as gym
from gymnasium import RewardWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print the name of your GPU

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.last_obs = None  # Store the last observation

    def step(self, action):
        # Perform a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update the last observation
        self.last_obs = obs

        # Modify the reward based on the observation
        reward = self.custom_reward(obs, reward)
        return obs, reward, terminated, truncated, info

    def custom_reward(self, obs, reward):
        # obs contains the state: [x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact]
        x, y, vx, vy, angle, angular_velocity, leg1, leg2 = obs
        
        # Example: Penalize for high descent rate
        if vy < -0.5:  # Negative velocity in the y direction
            reward -= 5
        # Bonus for being upright
        if abs(angle) < 0.1:
            reward += 1
        return reward
    
# Create environment
env = CustomRewardWrapper(gym.make('LunarLander-v2'))


# Instantiate the agent
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1,
    device='cuda')


# Train it for 1,000,000 timesteps
model.learn(total_timesteps=1000000)
# Save the model
model_name = "ppo-LunarLander-v2-custom-reward"
model.save(model_name)

eval_env = Monitor(CustomRewardWrapper(gym.make("LunarLander-v2", render_mode='rgb_array')))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
