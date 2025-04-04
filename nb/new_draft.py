import os
import time
import csv
import argparse
import gymnasium as gym
import openai

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global log to capture LLM-related reward prompts and outputs
llm_log = {
    "prompts": [],
    "llm_rewards": []
}

def call_llm_reward(prompt: str) -> float:
    """
    Calls the OpenAI ChatCompletion API with a given prompt and returns a float reward.
    The function logs the prompt and resulting reward.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        # Extract the text response and convert to float.
        text_response = response.choices[0].message.content.strip()
        new_reward = float(text_response)
        # Log the prompt and reward for later CSV logging.
        llm_log["prompts"].append(prompt)
        llm_log["llm_rewards"].append(new_reward)
        return new_reward
    except Exception as e:
        print(f"LLM API call failed: {e}")
        return 0.0  # fallback reward adjustment

class LLMRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that, on every step, calls the LLM API to re-evaluate the default reward.
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        prompt = (
            f"In the context of Lunar Lander, the default reward is {reward:.2f}. "
            "Rewrite this reward value to reflect a human-like evaluation. "
            "Return only a single numeric value."
        )
        new_reward = call_llm_reward(prompt)
        return new_reward

class SegmentedLLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, total_timesteps, segments=10):
        super().__init__(env)
        self.segments = segments
        self.total_timesteps = total_timesteps
        self.episode_cumulative = 0.0
        self.step_count = 0
        self.segment_length = max(1, total_timesteps // segments)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.episode_cumulative = 0.0
        self.step_count = 0
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self.episode_cumulative += reward
        self.step_count += 1

        new_reward = reward

        if (self.step_count % self.segment_length == 0) or done:
            prompt = (
                f"In the context of Lunar Lander over the last {self.step_count} steps, "
                f"the cumulative reward was {self.episode_cumulative:.2f}. "
                "Rewrite this cumulative reward into a refined numeric reward value that reflects performance. "
                "Return only a single numeric value."
            )
            new_reward = call_llm_reward(prompt)
            self.episode_cumulative = 0.0

        return observation, new_reward, done, truncated, info

def run_episode(env, episodes: int, max_timesteps: int):
    """
    Runs the environment for a specified number of episodes.
    Records every reward received during the run.
    """
    all_rewards = []  # store rewards from every step in every episode

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_steps = 0
        while not done and episode_steps < max_timesteps:
            action = env.action_space.sample()  # Replace with your agent's action selection
            obs, reward, done, truncated, info = env.step(action)
            all_rewards.append(reward)
            episode_steps += 1
            if done or truncated:
                break
    return all_rewards

def log_run_to_csv(run_mode: str, time_elapsed: float, rewards: list, reward_prompts: list, csv_file="runs.csv"):
    """
    Logs the run data to a CSV file.
    """
    fieldnames = ["run_mode", "time_elapsed", "rewards", "reward_prompts"]
    row = {
        "run_mode": run_mode,
        "time_elapsed": time_elapsed,
        "rewards": rewards,
        "reward_prompts": reward_prompts
    }
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write header only if the file did not exist before.
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def main(run_mode: str, total_timesteps: int, episodes: int):
    # Create the base environment
    env = gym.make("LunarLander-v2")

    # Wrap the environment based on the chosen mode
    if run_mode == "llm":
        print("Running in LLM reward mode: each step reward is rewritten by an LLM.")
        env = LLMRewardWrapper(env)
    elif run_mode == "segmented":
        print("Running in segmented LLM reward mode: rewards are rewritten per segment.")
        env = SegmentedLLMRewardWrapper(env, total_timesteps=total_timesteps, segments=10)
    else:
        print("Running in default mode with Gymnasium's original reward.")

    # Run and time the episodes
    start_time = time.time()
    rewards_collected = run_episode(env, episodes, total_timesteps)
    end_time = time.time()
    elapsed = end_time - start_time

    # Log the run data to CSV
    log_run_to_csv(run_mode, elapsed, rewards_collected, llm_log["prompts"])
    print(f"Run mode: {run_mode}\nTime Elapsed: {elapsed:.2f} seconds")
    print(f"Total Steps Recorded: {len(rewards_collected)}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lunar Lander RL reward evaluation with LLM-based reward rewriting")
    parser.add_argument("--mode", type=str, choices=["default", "llm", "segmented"], default="default",
                        help="Select the reward mode: 'default', 'llm', or 'segmented'.")
    parser.add_argument("--total_timesteps", type=int, default=1000,
                        help="Maximum timesteps per episode.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run.")
    args = parser.parse_args()
    main(args.mode, args.total_timesteps, args.episodes)
