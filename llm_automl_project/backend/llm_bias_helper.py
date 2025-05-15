import os
from llm_generator import generate_preprocessing_code


def bias_explainer(bias_metrics: dict) -> str:
    """
    Generate a natural language explanation of bias audit metrics using the LLM.

    Args:
        bias_metrics (dict): Dictionary of fairness metrics and their values.

    Returns:
        str: LLM-generated explanation text.
    """
    prompt = "Explain these bias audit results in plain language:\n"
    for metric, value in bias_metrics.items():
        prompt += f"{metric}: {value}\n"
    explanation = generate_preprocessing_code(prompt)
    return explanation


def inject_code_into_pipeline(task_description: str) -> str:
    """
    Generate Python code using the LLM and save it as a file.

    Args:
        task_description (str): Description of what the code should do.

    Returns:
        str: Path to the saved Python file.
    """
    code = generate_preprocessing_code(task_description)
    filepath = os.path.join("injected_pipeline_steps.py")

    with open(filepath, "w") as f:
        f.write(code)

    return filepath
