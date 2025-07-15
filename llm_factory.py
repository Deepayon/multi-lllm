import os
import json
import re
from models import gemini, gpt, claude, llama, deepseek

# Map string identifiers to LLM classes
LLM_DISPATCH = {
    "gemini": gemini.GeminiLLM,
    "gpt": gpt.GPTLLM,
    "openai": gpt.GPTLLM,
    "openrouter": gpt.GPTLLM,
    "claude": claude.ClaudeLLM,
    "llama": llama.LLaMALLM,
    "ollama": llama.LLaMALLM,
    "deepseek": deepseek.DeepSeekLLM
}


def generate_response(
    llm_type: str,
    data: dict,
    api_key: str = None,
    model: str = None,
    return_raw: bool = False
) -> str | dict:
    """
    Generic entrypoint to generate output from any LLM backend.

    Args:
        llm_type (str): One of ["gemini", "gpt", "claude", "llama", "deepseek", etc.]
        data (dict): Input dict to be formatted into a prompt by shared_prompt
        api_key (str): API key to use
        model (str): Optional model version
        return_raw (bool): If True, returns raw string output. Else parses JSON if possible.

    Returns:
        str or dict: LLM response (raw or parsed)
    """

    llm_type = llm_type.lower()
    if llm_type not in LLM_DISPATCH:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    try:
        llm_class = LLM_DISPATCH[llm_type]
        llm_instance = llm_class(model=model) if model else llm_class()

        output = llm_instance.generate(data, api_key)

        if return_raw:
            return output

        # Clean markdown-style ```json blocks
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", output).strip()

        # Try parsing as JSON
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")

    except Exception as e:
        raise RuntimeError(f"LLM processing failed: {e}")
