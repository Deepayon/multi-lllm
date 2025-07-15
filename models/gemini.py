# llm_sdk/models/gemini.py

import google.generativeai as genai
from llm_sdk.shared_prompt import build_prompt


class GeminiLLM:
    def __init__(self, model: str = "models/gemini-2.0-flash"):
        self.model = model

    def generate(self, access_counts: dict, api_key: str) -> str:
        if not access_counts:
            return "No access data provided."

        if not api_key:
            return "Gemini API error: API key is required but missing."

        try:
            genai.configure(api_key=api_key)
            prompt = build_prompt(access_counts)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)

            if not response or not hasattr(response, "text"):
                return "Gemini API returned an invalid response."

            return response.text.strip()

        except Exception as e:
            return f"Gemini API error: {e}"
