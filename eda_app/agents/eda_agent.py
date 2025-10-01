import os
import openai
import streamlit as st

gen = os.getenv("OPENAI_KEY_API")
if gen is None:
    gen = st.secrets["OPENAI_API_KEY"]

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
