def reward_function(obs, default_reward, previous_obs, action):
    if previous_obs is None:
        return default_reward
    
    if len(obs) >= 8:
        hull_y = obs[1]
        hull_vy = obs[3]
        leg_contact = obs[6]
        right_leg_contact = obs[7]
        
        # Reward for height above ground
        height_reward = 5.0 * (1 - abs(hull_y))
        
        # Penalty for vertical velocity
        velocity_penalty = -0.1 * abs(hull_vy)
        
        # Reward for landing stability
        stability_reward = 10.0 if leg_contact and right_leg_contact else 0.0
        
        # Combine rewards
        reward = height_reward + velocity_penalty + stability_reward - 0.03
        
        # Bound the reward
        return max(-100, min(100, reward))
    
    return default_reward