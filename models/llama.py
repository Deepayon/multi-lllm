# llm_sdk/models/llama.py

import os
import json
import re
import requests
from llm_sdk.shared_prompt import build_prompt


class LLaMALLM:
    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, access_counts: dict, api_key: str = None) -> str:
        if not access_counts:
            return "No access data provided."

        final_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not final_key:
            return "No API key provided."

        prompt = build_prompt(access_counts)

        headers = {
            "Authorization": f"Bearer {final_key}",
            "Content-Type": "application/json",
            "X-Title": "MultiLLM-SDK"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            if "choices" not in data or not data["choices"]:
                return f"OpenRouter returned invalid structure: {data}"

            raw = data["choices"][0]["message"]["content"]
            print(f"LLaMA raw response (first 100 chars): {repr(raw[:100])}")

            return self._extract_json(raw)

        except requests.RequestException as req_err:
            return f"LLaMA API error (HTTP): {req_err}"
        except ValueError as parse_err:
            return f"LLaMA response parsing error: {parse_err}"
        except Exception as e:
            return f"LLaMA unexpected error: {e}"

    def _extract_json(self, raw: str) -> str:
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/llm_raw_output.log", "w") as f:
                f.write(raw)
        except Exception as log_err:
            print(f"⚠️ Failed to write raw output to log: {log_err}")

        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw).strip()

        try:
            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON: {e}")
