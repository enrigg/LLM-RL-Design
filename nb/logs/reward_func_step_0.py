def reward_function(obs, default_reward, previous_obs, action):
    if previous_obs is None:
        return default_reward
    
    if len(obs) != 8:
        return default_reward
    
    hull_x, hull_y, hull_vx, hull_vy, hull_angle, hull_ang_vel, left_leg_contact, right_leg_contact = obs
    reward = -0.03  # Default per-step penalty
    
    # Reward for successful landing
    if left_leg_contact and right_leg_contact and hull_vy <= 0.5:
        reward += 100  # Successful landing
    
    # Penalty for crashing
    elif hull_vy > 1.0 or abs(hull_vx) > 1.0:
        reward -= 100  # Crash penalty
    
    # Reward for reducing vertical velocity
    if hull_vy < 0:
        reward += min(1.0, -hull_vy)  # Encourage slowing down
    
    # Reward for moving towards the landing zone
    if hull_y < 0:
        reward += min(1.0, -hull_y)  # Encourage descending towards the surface
    
    # Penalize excessive angular velocity
    if abs(hull_ang_vel) > 1.0:
        reward -= 0.5 * abs(hull_ang_vel)
    
    # Bound the reward to [-100, 100]
    return max(-100, min(100, reward))