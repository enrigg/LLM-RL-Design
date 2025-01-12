import openai

openai.api_key = ""

def openai_call():
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "developer", 
                "content": """You are an expert on python, especially in reward function, your job is to return a function, and just the function and it must be called custom_reward()
                Make sure it returns a reward. And dont include ```python at the beginning or ``` at the end.
                the function should be an easy one like:
                2+2
                """
            }
        ]
    )
    function_code = completion.choices[0].message.content
    return function_code


# Get the function code from OpenAI
function_code = openai_call()

# Print the generated function
print(function_code)

exec(function_code)

result = custom_reward()
print(result)