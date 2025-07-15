import os
import json
import re
import requests
from shared_prompt import build_prompt


class DeepSeekLLM:
    def __init__(self, model: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"):
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, access_counts: dict, api_key: str = None) -> str:
        if not access_counts:
            return "No access data provided."

        prompt = build_prompt(access_counts)
        final_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not final_key:
            return "No API key provided."

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

            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not raw:
                return "DeepSeek response did not include valid message content."

            print(f"DeepSeek model used: {self.model}")
            print(repr(raw))  # Debug print

            return self._extract_json(raw)

        except requests.RequestException as req_err:
            return f"DeepSeek API error (HTTP): {req_err}"
        except ValueError as parse_err:
            return f"DeepSeek response parsing error: {parse_err}"
        except Exception as e:
            return f"DeepSeek unexpected error: {e}"

    def _extract_json(self, raw: str) -> str:
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw).strip()
        try:
            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON: {e}")
