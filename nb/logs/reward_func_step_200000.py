def reward_function(obs, default_reward, previous_obs, action):
    if previous_obs is None:
        return default_reward
    
    if len(obs) < 8:
        return default_reward
    
    hull_y = obs[1]
    hull_vy = obs[3]
    leg_contact = obs[6]
    right_leg_contact = obs[7]
    
    reward = -0.03  # Per-step penalty
    
    # Reward for height above ground
    if hull_y > 0:
        reward += 5.0 * (1 - abs(hull_y))
    
    # Reward for low vertical velocity
    if hull_vy < 1:
        reward += 2.0 * (1 - abs(hull_vy))
    
    # Reward for landing stability
    if leg_contact and right_leg_contact:
        reward += 100.0  # Successful landing
    
    # Penalty for high vertical velocity
    if hull_vy < -3:
        reward -= 50.0  # Too fast descent
    
    # Penalty for excessive angular velocity
    if abs(obs[5]) > 1:
        reward -= 10.0  # High angular velocity penalty
    
    # Bound the reward to [-100, 100]
    return max(-100, min(100, reward))