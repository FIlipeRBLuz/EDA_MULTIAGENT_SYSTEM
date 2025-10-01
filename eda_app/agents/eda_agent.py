import os
import openai

gen = os.getenv("OPENAI_KEY_API")

openai.api_key = gen

def query_agent(user_input, system_prompt="Você é um assistente útil."):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message["content"]
