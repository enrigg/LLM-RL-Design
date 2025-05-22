import gymnasium as gym
import numpy as np
from llm_interface import query_llm
from prompts import REWARD_FUNCTION_PROMPT
import re
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, strategy='static', refresh_interval=None):
        super().__init__(env)
        self.strategy = strategy
        self.previous_obs = None
        self.step_counter = 0
        self.refresh_interval = refresh_interval or 1_000_000
        self.max_dynamic_updates = 10
        self.dynamic_update_count = 0
        self.previous_reward_functions = []
        self.reward_results = []
        self.current_dynamic_reward_func = None

        if self.strategy == 'static':
            self._init_static_reward()

    def _init_static_reward(self):
        raw_code = query_llm(REWARD_FUNCTION_PROMPT.format(history_text=""))
        self.reward_func_code = self._clean_code(raw_code)
        try:
            local_vars = {}
            exec(self.reward_func_code, {}, local_vars)
            self.reward_func = local_vars.get("reward_function", None)

            if self.reward_func:
                # Test with multiple sample observations
                for _ in range(3):
                    sample_obs = self.env.observation_space.sample()
                    sample_action = self.env.action_space.sample()
                    try:
                        test_reward = self.reward_func(sample_obs, 0.0, None, sample_action)
                        if not isinstance(test_reward, (int, float)):
                            logger.error(f"Validation failed for static reward function: Returned {test_reward} (type: {type(test_reward)})")
                            raise ValueError("Reward function did not return a single float")
                        logger.info("Static reward function validated successfully")
                    except Exception as e:
                        logger.error(f"Validation failed for static reward function: {e}")
                        self.reward_func = None
                        break

            os.makedirs("logs", exist_ok=True)
            with open("logs/reward_func_static.py", "w") as f:
                f.write(self.reward_func_code)
        except Exception as e:
            logger.error(f"Error executing static reward function: {e}")
            self.reward_func = None

    def _clean_code(self, code: str) -> str:
        cleaned = re.sub(r"```(?:python)?", "", code)
        return cleaned.strip("`\n ")

    def _extract_number(self, text: str) -> float:
        cleaned = re.sub(r"```(?:python)?", "", text)
        cleaned = cleaned.strip("`\n ")
        try:
            return float(cleaned)
        except ValueError:
            raise ValueError(f"Could not convert to float: '{cleaned}'")

    def reset(self, **kwargs):
        self.previous_obs, _ = self.env.reset(**kwargs)
        return self.previous_obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.strategy == 'static' and self.reward_func:
            try:
                reward = self.reward_func(obs, reward, self.previous_obs, action)
            except Exception as e:
                logger.error(f"Error applying static reward function: {e}")
                reward = reward  # Fallback to default reward

        elif self.strategy == 'dynamic':
            if self.step_counter % self.refresh_interval == 0 and self.dynamic_update_count < self.max_dynamic_updates:
                history_text = "\n\n".join(
                    f"# Reward Function {i+1}:\n{func}\n# Outcome:\nObs: {out[0]}\nAction: {out[1]}\nNew Obs: {out[2]}\nDefault Reward: {out[3]}\nModified Reward: {out[4]}"
                    for i, (func, out) in enumerate(zip(self.previous_reward_functions, self.reward_results))
                )
                prompt = REWARD_FUNCTION_PROMPT.format(history_text=history_text)

                try:
                    raw_code = query_llm(prompt)
                    cleaned_code = self._clean_code(raw_code)
                    local_vars = {}
                    exec(cleaned_code, {}, local_vars)
                    reward_func = local_vars.get("reward_function", None)

                    if reward_func:
                        # Test with multiple sample observations
                        for _ in range(3):
                            sample_obs = self.env.observation_space.sample()
                            sample_action = self.env.action_space.sample()
                            try:
                                test_reward = reward_func(sample_obs, 0.0, None, sample_action)
                                if not isinstance(test_reward, (int, float)):
                                    logger.error(f"Validation failed for dynamic reward function at step {self.step_counter}: Returned {test_reward} (type: {type(test_reward)})")
                                    raise ValueError("Reward function did not return a single float")
                                logger.info(f"Dynamic reward function validated successfully at step {self.step_counter}")
                            except Exception as e:
                                logger.error(f"Validation failed for dynamic reward function at step {self.step_counter}: {e}")
                                reward_func = None
                                break

                        if reward_func:
                            self.current_dynamic_reward_func = reward_func
                            self.previous_reward_functions.append(cleaned_code)
                            self.dynamic_update_count += 1

                            os.makedirs("logs", exist_ok=True)
                            with open(f"logs/reward_func_step_{self.step_counter}.py", "w") as f:
                                f.write(cleaned_code)

                except Exception as e:
                    logger.error(f"Error generating dynamic reward function at step {self.step_counter}: {e}")

            if self.current_dynamic_reward_func:
                try:
                    new_reward = self.current_dynamic_reward_func(obs, reward, self.previous_obs, action)
                    self.reward_results.append((self.previous_obs, action, obs, reward, new_reward))
                    reward = new_reward
                except Exception as e:
                    logger.error(f"Error applying current dynamic reward function: {e}")
                    reward = reward  # Fallback to default reward

        self.previous_obs = obs
        self.step_counter += 1
        return obs, reward, terminated, truncated, info