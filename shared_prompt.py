def build_prompt(data: dict, task_description: str) -> str:
    """
    Builds a strict LLM prompt for parsing structured key → value output.

    Args:
        data (dict): Input dictionary to be processed by the LLM.
        task_description (str): Clear instruction on what the LLM should do with the data.

    Returns:
        str: A strict, reusable LLM prompt string.
    """

    if not task_description:
        raise ValueError("Prompt requires a task_description.")

    prompt = (
        f"{task_description.strip()}\n\n"
        "Output Requirements:\n"
        "- Return only key-value pairs (one per line).\n"
        "- Format: <KEY> => <VALUE>\n"
        "- Do NOT include explanations, markdown, comments, or conversational text.\n"
        "- Do NOT wrap the output in JSON or any block formatting.\n"
        "- Ensure output is clean and directly parseable line-by-line.\n\n"
        "=== Input ===\n"
    )

    for key, value in sorted(data.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
        prompt += f"{key}: {value}\n"

    prompt += "\n=== Begin Output ===\n"
    return prompt
