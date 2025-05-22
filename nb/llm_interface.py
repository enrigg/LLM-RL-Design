from openai import OpenAI
from prompts import REWARD_FUNCTION_PROMPT

client = OpenAI()

def query_llm(prompt: str) -> str:
    final_prompt = REWARD_FUNCTION_PROMPT.format(history_text=prompt) if "{history_text}" in prompt else prompt

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a reinforcement learning assistant. Reply only with the Python code of the reward function, no explanations or markdown.\n"
                    "Write a function called `reward_function(obs, default_reward, previous_obs, action)` that improves the reward design for the specified Gymnasium environment.\n"
                    "The function must:\n"
                    "- Return a single float (e.g., 0.0). Do NOT return lists, tuples, arrays, or None.\n"
                    "- Use safe indexing for obs and previous_obs (e.g., obs[0] for vectors, obs['key'] for dictionaries) based on the observation space structure.\n"
                    "- Avoid hard-coded unpacking that assumes a fixed number of observation components.\n"
                    "- Check observation length or structure to prevent unpacking errors (e.g., use len(obs) or try-except).\n"
                    "- Avoid external libraries (use only built-in Python math).\n"
                    "- Handle edge cases (e.g., previous_obs is None, complex observation spaces).\n"
                    "- Ensure numerical stability (e.g., avoid division by zero, bound outputs to [-100, 100]).\n"
                    "- Align with the environment's goal, avoid sparse or exploitable rewards, and ensure compatibility with RL algorithms.\n"
                )
            },
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_prompt(entorno: str, output_filename: str = "prompts.py") -> None:
    print(f"Generating prompt for {entorno}")
    prompt = f"""
        You are tasked with creating a detailed description for the Gymnasium environment "{entorno}" to guide the generation of a custom reward function. The description will be used by an LLM to produce a Python function `reward_function(obs, default_reward, previous_obs, action)` that returns a single float. Consult the official Gymnasium documentation for "{entorno}" to ensure accuracy and completeness.

        Include the following sections in the output:

        ### 1. Environment Context
        - **Purpose**: Describe the environment's objective (e.g., navigation, balancing, control, game-playing).
        - **Observation Space**: Specify the type (e.g., Box, Dict, Tuple), shape (e.g., Box(24,), Box(96,96,3)), and list all components with:
          - Name (e.g., hull x-position, joint angle, lidar reading).
          - Physical meaning (e.g., robot's horizontal position, knee joint angle, distance to terrain).
          - Units (if applicable, e.g., meters, radians, pixels).
          - Range or bounds (e.g., [-1, 1], {0, 1} for booleans, [0, 255] for images).
          - Exact number of components for vector spaces (e.g., 24 for BipedalWalker-v3, 8 for LunarLander-v2).
          - For complex spaces (e.g., Dict, Tuple, Box with high dimensions), describe each sub-component recursively with access method (e.g., obs['key'], obs[0][1]).
        - **Action Space**: Specify the type (e.g., Discrete, Box, MultiDiscrete) and describe all possible actions:
          - For Discrete, list each action and its effect (e.g., 0: no-op, 1: move left).
          - For Box, list each dimension, its bounds (e.g., [-1, 1]), and effect (e.g., torque on hip joint).
          - For MultiDiscrete or complex spaces, describe each component and its effect.
        - **Starting State**: Describe the initial state (e.g., position, velocity, randomization, or fixed state).
        - **Termination Conditions**: List all conditions for episode termination (e.g., timeout, crash, goal reached, out-of-bounds).
        - **Success/Failure**: Define success (e.g., reaching a goal state, achieving a reward threshold) and failure (e.g., crashing, timeout) with measurable criteria.

        ### 2. Goal
        - **Objective**: State the agent's primary goal (e.g., "walk across terrain", "balance a pole", "maximize score").
        - **Success Conditions**: Specify measurable conditions for success (e.g., "distance traveled ≥ 200 meters", "upright for 200 steps").
        - **Avoidance**: List states or behaviors to avoid (e.g., falling, excessive speed, instability).
        - **Progress Metrics**: Suggest measurable indicators of progress (e.g., distance traveled, stability, score increase).

        ### 3. Reward Guidelines
        - **Default Reward**: Provide the exact reward structure from the documentation, including:
          - Per-step rewards/penalties (e.g., based on distance, speed, actions, state variables).
          - Terminal rewards (e.g., -100 for falling, +100 for success).
          - Solution threshold, if any (e.g., total episode reward ≥ 300 for success).
          - Any additional reward mechanics (e.g., penalties for motor torque, bonuses for milestones).
        - **Weaknesses**: Identify limitations in the default reward, such as:
          - Sparse rewards that slow learning.
          - Encouragement of risky or suboptimal behaviors (e.g., excessive speed).
          - Lack of progress signals for early training.
          - Over-penalization of actions.
        - **Improvements**: Suggest enhancements, including:
          - Dense rewards for partial progress (e.g., reward based on distance traveled, stability).
          - Penalties for undesired states (e.g., high angular velocity, falling).
          - Smoothing rewards to avoid abrupt changes (e.g., exponential decay for penalties).
          - Normalizing rewards to a range (e.g., [-100, 100]) for RL algorithm stability.
        - **Constraints**: Ensure the reward function:
          - Returns a single float.
          - Uses safe indexing for observations (e.g., obs[0] for vectors, obs['key'] for dictionaries) based on the documented observation space.
          - Avoids hard-coded unpacking that assumes a fixed number of components (e.g., do not assume obs has exactly 8 values).
          - Checks observation length or structure to prevent unpacking errors (e.g., use len(obs) or try-except).
          - Avoids sparse rewards that hinder learning.
          - Prevents reward hacking (e.g., exploiting stationary states to avoid penalties).
          - Maintains numerical stability (e.g., bounds outputs, avoids division by zero).
        - Make sure it rewards the moves needed to get to the goal, and penalizes the moves that dont (like going up in the LunarLander or going backward in the bidedalWalker)

        ### 4. For Dynamic Strategy (History-Based Updates)
        - If history is provided (via {{history_text}}):
          - Analyze previous reward functions and outcomes (e.g., observation, action, default reward, modified reward).
          - Identify issues, such as:
            - Over-rewarding a variable (e.g., height causing oscillation or stalling).
            - Under-rewarding progress (e.g., ignoring distance traveled).
            - Unstable outputs (e.g., extreme values causing divergence).
            - Failure to meet solution threshold (e.g., total reward below expected value).
            - Observation unpacking errors due to mismatched component counts.
          - Suggest conservative adjustments, e.g.:
            - Adjust weights if a variable dominates (e.g., reduce height reward if agent stalls).
            - Add penalties for undesired behaviors (e.g., high angular velocity, falling).
            - Fix unpacking errors by using dynamic indexing or checking obs length.
            - Smooth rewards if outcomes show erratic behavior.
          - Example: If history shows unpacking errors (e.g., 'too many values to unpack'), use len(obs) to verify component count or access obs safely.
        - Prioritize smooth, goal-directed behavior aligned with the environment’s objective and solution threshold.

        ### 5. Additional Constraints
        - The reward function must:
          - Handle edge cases (e.g., previous_obs is None on the first step, complex observation spaces like images or dictionaries).
          - Use only built-in Python math (no numpy, scipy, etc.).
          - Ensure numerical stability (e.g., use min/max to bound values, avoid division by zero).
          - Return a float in a reasonable range (e.g., [-100, 100]) to avoid RL algorithm issues.
          - Verify observation structure before accessing components (e.g., check len(obs) or use try-except for unpacking).
        - Provide example calculations for key states (e.g., near goal, far from goal, failure state) to guide the LLM, including expected reward values and safe observation access.
        - Example code for safe observation handling (for a vector observation space):
          ```python
          if len(obs) >= 24:  # For BipedalWalker-v3
              hull_x, hull_y, hull_vx, hull_vy, hull_angle, hull_ang_vel = obs[0:6]
              reward = 5.0 * hull_vx  # Reward forward progress
          else:
              reward = default_reward  # Fallback
          ```
        - Example for complex spaces (e.g., Dict):
          ```python
          if 'hull_position' in obs:
              hull_x = obs['hull_position'][0]
              reward = 5.0 * hull_x  # Reward forward progress
          else:
              reward = default_reward  # Fallback
          ```

        Return the description as plain text to guide the reward function’s creation, not the function itself. End with this exact line:
        # Previous History (for dynamic strategy):
        {{history_text}}
        """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=3500  # Increased for detailed output
    )
    print("promt generado")
    result_code = response.choices[0].message.content
    print(f"guardando como {output_filename}")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("REWARD_FUNCTION_PROMPT = '''\\\n")
        f.write(result_code)
        f.write("\n'''")