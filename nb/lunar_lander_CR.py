import gymnasium as gym
from gymnasium import RewardWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch
from reward_shaping import custom_reward_class
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

reward_generator = custom_reward_class()
reward_generator.generate_function() 

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, custom_reward_func):
        super(CustomRewardWrapper, self).__init__(env)
        self.custom_reward_func = custom_reward_func
        self.last_obs = None  # Store the last observation

    def step(self, action):
        # Perform a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update the last observation
        self.last_obs = obs

        # Modify the reward based on the observation
        reward = self.custom_reward_func(obs, reward)
        return obs, reward, terminated, truncated, info

    
# Create environment
env = CustomRewardWrapper(
                        gym.make('LunarLander-v2'),
                        custom_reward_func=reward_generator.generated_function)


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
model.learn(total_timesteps=10000)
# Save the model
model_name = "ppo-LunarLander-v2-custom-reward"
model.save(model_name)

eval_env = Monitor(
    CustomRewardWrapper(
        gym.make("LunarLander-v2", render_mode="rgb_array"),
        custom_reward_func=reward_generator.generated_function
    )
)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
