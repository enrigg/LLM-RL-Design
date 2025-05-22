REWARD_FUNCTION_PROMPT = '''\
### 1. Environment Context
- **Purpose**: The "BipedalWalker-v3" environment simulates a bipedal robot navigating across varied terrain. The objective is to control the robot's movement in a way that allows it to walk forward without falling over or losing balance.
- **Observation Space**: The observation space is a continuous Box with shape (24,). The components of the observation include:
  - Name: hull x-position, Physical Meaning: robot's horizontal position, Units: meters, Range: [-1e6, 1e6].
  - Name: hull y-position, Physical Meaning: robot's vertical position, Units: meters, Range: [-1e6, 1e6].
  - Name: hull velocity x, Physical Meaning: robot's horizontal velocity, Units: meters/second, Range: [-1e6, 1e6].
  - Name: hull velocity y, Physical Meaning: robot's vertical velocity, Units: meters/second, Range: [-1e6, 1e6].
  - Name: hull angle, Physical Meaning: robot's orientation angle, Units: radians, Range: [-π, π].
  - Name: hull angular velocity, Physical Meaning: robot's rotational speed, Units: radians/second, Range: [-1e6, 1e6].
  - Name: joint angles (8 total), Physical Meaning: angles of the robot's joints, Units: radians, Range: [-π, π].
  - Name: joint angular velocities (8 total), Physical Meaning: rotational speeds of the robot's joints, Units: radians/second, Range: [-1e6, 1e6].
- **Action Space**: The action space is a continuous Box with shape (4,). The actions correspond to torques applied to the robot's joints:
  - Action 1: torque on left hip, Bounds: [-1, 1].
  - Action 2: torque on right hip, Bounds: [-1, 1].
  - Action 3: torque on left knee, Bounds: [-1, 1].
  - Action 4: torque on right knee, Bounds: [-1, 1].
- **Starting State**: The robot starts in a random position and orientation, with initial velocities set to zero.
- **Termination Conditions**: The episode terminates if:
  - The robot falls over (hull y-position < 0).
  - The robot travels a specified distance (goal) or exceeds a time limit (e.g., 2000 steps).
- **Success/Failure**: Success is defined by the robot traveling a distance of at least 200 meters without falling over. Failure occurs if the robot falls or does not complete the task within the time limit.

### 2. Goal
- **Objective**: The primary goal of the agent is to walk across the terrain effectively without falling over.
- **Success Conditions**: The measurable condition for success is traveling a distance of at least 200 meters while remaining upright for the duration of the episode.
- **Avoidance**: The agent should avoid states where it falls over, which is indicated by a hull y-position less than zero, as well as excessive angular velocities that could lead to instability.
- **Progress Metrics**: Key indicators of progress include the distance traveled (measured in meters) and the stability of the robot (measured by maintaining an upright position).

### 3. Reward Guidelines
- **Default Reward**: The default reward structure includes:
  - Per-step reward: +0.1 for each time step the robot remains upright and does not fall.
  - Terminal rewards: -100 for falling over (hull y-position < 0), +100 for successfully traveling the goal distance of 200 meters.
  - Solution threshold: Total episode reward ≥ 300 for success.
  - Additional mechanics: Penalties may apply for excessive torque use, which might be implemented as a small negative reward based on the absolute values of the torques applied.
- **Weaknesses**: Limitations in the default reward structure include:
  - Sparse rewards that can slow down learning, especially early on.
  - Potential encouragement of risky behaviors, such as excessive speed or torque application.
  - Lack of immediate feedback for progress, making it difficult for the agent to learn effectively.
  - Over-penalization of certain actions that may not lead to significant improvements.
- **Improvements**: Suggested enhancements for the reward function include:
  - Providing dense rewards based on distance traveled, with incremental rewards for every meter moved forward.
  - Introducing penalties for high angular velocities or unstable positions, preventing falls.
  - Smoothing rewards to avoid abrupt changes, perhaps using a decay factor for penalties.
  - Normalizing the total reward to a range of [-100, 100] to ensure stability in reinforcement learning algorithms.
- **Constraints**: The reward function must:
  - Return a single float.
  - Use safe indexing for observations (e.g., `obs[0]` for vectors).
  - Avoid hard-coded unpacking that assumes a fixed number of components.
  - Check observation length or structure to prevent unpacking errors.
  - Avoid sparse rewards that hinder learning.
  - Prevent reward hacking by ensuring that rewards are not easily exploitable.
  - Maintain numerical stability by bounding outputs and avoiding division by zero.

### 4. For Dynamic Strategy (History-Based Updates)
- If history is provided:
  - Analyze previous reward functions and outcomes to identify issues such as over-rewarding certain variables or under-rewarding progress.
  - Suggest conservative adjustments, such as reducing the reward for height if it leads to oscillation or stalling.
  - Add penalties for undesired behaviors like high angular velocity or falling.
  - Fix unpacking errors by using dynamic indexing or checking observation lengths.
  - Prioritize smooth, goal-directed behavior aligned with the environment’s objectives.

### 5. Additional Constraints
- The reward function must:
  - Handle edge cases (e.g., when `previous_obs` is None on the first step).
  - Use only built-in Python math (no external libraries).
  - Ensure numerical stability (e.g., use min/max to bound values).
  - Return a float in a reasonable range (e.g., [-100, 100]).
  - Verify observation structure before accessing components.
- Example calculations:
  - For a state where the robot is upright and moving forward: `reward = 0.1 + (distance_traveled * 0.1)`.
  - For a state where the robot falls: `reward = -100`.
- Example code for safe observation handling:
  ```python
  if len(obs) >= 24:  # For BipedalWalker-v3
      hull_x, hull_y, hull_vx, hull_vy, hull_angle, hull_ang_vel = obs[0:6]
      reward = 5.0 * hull_vx  # Reward forward progress
  else:
      reward = default_reward  # Fallback
  ```

# Previous History (for dynamic strategy):
{history_text}
'''