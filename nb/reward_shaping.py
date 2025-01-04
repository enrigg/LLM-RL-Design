import openai


def openai_call():
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "developer", 
                "content": """You are an expert on Reinforcement learning, especially in reward function, your job is to return a function, and just the function and it must be called custom_reward() with the following parameters (obs, reward)
                Make sure it returns a reward. And dont include ```python at the beginning or ``` at the end.
                An example is the following:
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
                The user will provide context on the environment and based on that context and the feedback from the environment, you need to create an appropiate function"""
            },
            {
                "role": "user",
                "content": "The environment is Lunar Lander from the gymnasium."
            }
        ]
    )
    function_code = completion.choices[0].message.content
    return function_code


class custom_reward_class:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.role = "developer"
        self.content = """You are an expert on Reinforcement learning, especially in reward function, your job is to return a function, 
                and just the function and it must be called custom_reward() with the following parameters (obs, reward)
                Make sure it returns a reward. And dont include ```python at the beginning or ``` at the end.
                An example is the following:
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
                The user will provide context on the environment and based on that context and the feedback from the environment, 
                you need to create an appropiate function
                """
        self.review = 1000
    def generate_function(self):
        """Fetch and dynamically execute the custom reward function."""
        completion = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": self.role, "content": self.content},
                {"role": "user", "content": "The environment is Lunar Lander from the gymnasium."}
            ]
        )
        function_code = completion.choices[0].message.content.strip()
        exec(function_code)  # Dynamically execute the function code
        self.generated_function = locals()['custom_reward']  # Store the function for reuse

    def apply(self, obs, reward):
        """Apply the generated custom reward function."""
        if self.generated_function is None:
            raise RuntimeError("The reward function has not been generated yet. Call generate_function() first.")
        return self.generated_function(obs, reward)

custom_reward = custom_reward_class()
print(custom_reward.role)
#new_function = custom_reward.apply()
#print(new_function)