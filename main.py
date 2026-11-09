import sys
import json
import re
import argparse
import ollama

# Model configuration with auto-detection for instructor's grading system
def get_target_model() -> str:
    """Detects available model: prefers instructor's 27b model, falls back to local 7b model."""
    try:
        models = ollama.list()
        model_names = []
        if isinstance(models, dict):
            model_names = [m.get("model", "") for m in models.get("models", [])]
        else:
            model_names = [m.model for m in models.models]
            
        # Check if instructor's model is available
        for name in model_names:
            if "qwen3.8" in name.lower() or "27b" in name.lower():
                return name
    except Exception:
        pass
    
    # Default fallback to your local testing model
    return "qwen2.5-coder:7b"

TARGET_MODEL = get_target_model()


def extract_clean_answer(text: str) -> str:
    """Extracts only the final minimal answer, stripping explanations and markdown."""
    text = text.strip()
    
    # 1. Check for standard "Final Answer:" pattern
    match = re.findall(r"(?:Final\s+Answer|Answer)\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if match:
        candidate = match[-1].strip()
    else:
        # 2. Take the last non-empty line
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidate = lines[-1] if lines else text

    # 3. Strip markdown, surrounding quotes, and common wrappers
    cleaned = re.sub(r"^[\*`\"'#\s]+|[\*`\"'\s]+$", "", candidate).strip()
    return cleaned if cleaned else text.strip()


def solve_problem(problem_text: str, model_name: str = TARGET_MODEL) -> str:
    """Executes Chain-of-Thought reasoning locally and returns only the final answer."""
    system_prompt = (
        "You are an expert problem-solving agent. "
        "Solve the given problem step-by-step internally. "
        "At the very end of your response, output exactly in this format:\n"
        "Final Answer: <only the final value or required symbol>"
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem_text}
            ],
            options={"temperature": 0.0}
        )
        raw_output = response["message"]["content"]
        return extract_clean_answer(raw_output)

    except Exception:
        # Fallback to keep JSON format valid even if an error occurs (as required by the assignment)
        return "ERROR"


def main():
    parser = argparse.ArgumentParser(description="LLM Agent Evaluation Runner")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file.")
    args = parser.parse_args()

    # Read input file
    with open(args.input, "r", encoding="utf-8") as fin:
        input_data = json.load(fin)

    # Handle both single-object input {"id": ..., "problem": ...} and list input [...]
    is_list = isinstance(input_data, list)
    tasks = input_data if is_list else [input_data]

    results = []
    for item in tasks:
        task_id = item.get("id", "")
        problem_text = item.get("problem", "")
        answer = solve_problem(problem_text)
        results.append({
            "id": task_id,
            "answer": answer
        })

    # Output matches the input structure
    final_output = results if is_list else results[0]

    # Write output JSON
    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(final_output, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()