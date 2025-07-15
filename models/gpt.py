import os
import json
import re
import openai
from llm_sdk.shared_prompt import build_prompt


class GPTLLM:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    def generate(self, access_counts: dict, api_key: str = None) -> str:
        if not access_counts:
            return "No access data provided."

        final_key = api_key or os.getenv("OPENAI_API_KEY")
        if not final_key:
            return "OpenAI API key is required."

        openai.api_key = final_key
        prompt = build_prompt(access_counts)

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            raw = response["choices"][0]["message"]["content"].strip()

            # Optional: log raw output
            try:
                os.makedirs("logs", exist_ok=True)
                with open("logs/llm_raw_output.log", "w") as f:
                    f.write(raw)
            except Exception as log_err:
                print(f"Warning: Failed to log raw output – {log_err}")

            return self._extract_json(raw)

        except openai.error.OpenAIError as e:
            return f"OpenAI API error: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

    def _extract_json(self, raw: str) -> str:
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw).strip()
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}')
        if json_start != -1 and json_end != -1:
            cleaned = cleaned[json_start:json_end + 1]

        try:
            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI response did not return valid JSON: {e}")
