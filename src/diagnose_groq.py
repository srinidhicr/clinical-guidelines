"""Diagnose available models for the configured GROQ_API_KEY."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ_API_KEY is not set in .env")
else:
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        models = [m.id for m in client.models.list().data]
        print("Available Groq models on your account:")
        for m in sorted(models):
            print(f"  - {m}")
    except Exception as e:
        print(f"Error checking Groq models: {e}")
