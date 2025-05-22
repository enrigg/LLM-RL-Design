REWARD_FUNCTION_PROMPT = """
Write a Python function called `reward_function(obs, default_reward, previous_obs, action)` for the LunarLander-v2 environment in OpenAI Gym.
The function must return a single scalar float reward (not a list or array).
Do not use any external libraries.

### Environment Context
In LunarLander-v2, the lander starts each episode at a high altitude (obs[1] typically around 1.0 to 1.4) with a random x-position (obs[0] around -0.5 to 0.5) and near-zero initial velocity (obs[2], obs[3]). Gravity pulls the lander downward, reducing obs[1] unless the main engine (action=2) or side engines (action=1, 3) are used. The lander must descend to the landing pad at coordinates (0,0) and land softly with both legs touching the ground (obs[6] = 1, obs[7] = 1), near-zero velocity, and an upright orientation (obs[4] close to 0). Crashing (high velocity or incorrect orientation on ground contact) or moving outside the environment boundaries results in episode termination with a large negative default reward (-100).

The observation vector `obs` contains:
- obs[0]: x-coordinate (positive right, negative left, landing pad at x=0)
- obs[1]: y-coordinate (positive up, landing pad at y=0)
- obs[2]: x-velocity (positive right)
- obs[3]: y-velocity (positive up)
- obs[4]: angle (radians, 0 is upright, positive clockwise)
- obs[5]: angular velocity (radians/second)
- obs[6]: left leg contact (1 if touching ground, 0 otherwise)
- obs[7]: right leg contact (1 if touching ground, 0 otherwise)

The action space is discrete:
- action=0: do nothing
- action=1: fire left orientation engine
- action=2: fire main engine (upward thrust)
- action=3: fire right orientation engine

### Goal
The goal is to guide the lander to descend from its starting position to the landing pad at (0,0) immediately from the episode’s start and land with:
- Both legs touching the ground (obs[6] = 1 and obs[7] = 1), critical for landing.
- Near-zero velocity (abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2), critical for a soft landing to avoid crashing.
- Small x and y distance from (0,0) (abs(obs[0]) < 0.1 and abs(obs[1]) < 0.15).
- Upright orientation (abs(obs[4]) < 0.1), critical to prevent upside-down crashes.
- If landed successfully, stop the episode.
- Avoid crashing (high velocity or incorrect orientation on ground contact) or staying at high altitude.

### Reward Guidelines
1. Give a one-time large positive reward (e.g., +200) for landing on the pad, defined as:
   - abs(obs[0]) < 0.1 (close to x=0).
   - abs(obs[1]) < 0.15 (close to y=0).
   - abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2 (low velocity).
   - obs[6] = 1 and obs[7] = 1 (both legs on ground).
   - Add this reward to other terms; do not return it immediately.
2. Reward controlled descent: provide a small positive reward proportional to the reduction in y-coordinate compared to `previous_obs` (e.g., +2 * (prev_y - obs[1]), capped at +5 to avoid rapid descent).
3. Penalize high altitude: if obs[1] > 0.2 and no leg contact (obs[6] = 0 and obs[7] = 0), apply a very large negative reward (e.g., -100).
4. Penalize upward velocity: if obs[3] > 0, apply a large negative reward (e.g., -20 * obs[3]).
5. Penalize downward velocity: if obs[3] < 0, apply a very large negative reward (e.g., -15 * abs(obs[3])) to ensure a soft landing and prevent crashing.
6. Penalize distance from x=0 using abs(obs[0]) (e.g., -3 * abs(obs[0])).
7. Penalize lateral velocity using abs(obs[2]) (e.g., -2 * abs(obs[2])).
8. Penalize non-upright orientation using abs(obs[4]) (e.g., -10 * abs(obs[4])) to ensure the lander remains upright and avoid upside-down crashes.
9. Penalize high angular velocity using abs(obs[5]) (e.g., -2 * abs(obs[5])).
10. Optionally use `default_reward` to incorporate the environment's default reward (e.g., scale by 0.1 to minimize crash penalty impact).
11. Use `action` to penalize excessive control effort (e.g., -0.1 for non-zero actions).
12. Ensure the reward is bounded (e.g., clamp between -100 and 100) to avoid extreme values.
13. Note: The environment will terminate the episode on a successful landing, so focus on guiding the lander to descend softly and land upright in the proper coordinates from the episode’s start.

### For Dynamic Strategy (if history is provided)
- Analyze previous reward functions and outcomes in the history.
- Identify and fix issues, such as over-rewarding high altitude, over-rewarding rapid descent, under-penalizing upward or downward velocity, under-penalizing non-upright orientation, under-penalizing lateral movement, or rewarding actions after landing.
- Prioritize controlled descent from the episode’s start, upright orientation (abs(obs[4]) close to 0), and a soft landing with near-zero velocity and leg contact, strongly discouraging climbing, rapid descent, or upside-down crashes.
- If the reward has good momentum, don’t make big changes, only if needed for improving results.

Return only the Python code for the reward function.
The landing pad is always at coordinates (0,0).
# Previous History (for dynamic strategy):
{history_text}
"""