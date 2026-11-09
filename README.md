# LLM Reasoning Agent - Assignment 1

Personal implementation of the reasoning agent framework designed for the Intelligent Systems course assignment. This agent operates entirely offline using local LLM inference via Ollama, satisfying all air-gapped and runtime constraints.

---

## 🛠️ Engineering Highlights & Core Design

Unlike standard wrapper scripts, this implementation focuses on resilience and adaptability:
- **Zero-Cloud Dependency**: Fully localized execution pipeline (`http://localhost:11434`). No API keys, external networks, or cloud runtimes required.
- **Smart Model Discovery (`get_target_model`)**: Automatically queries the local Ollama daemon at runtime. It searches for the primary evaluation model (`qwen3.8:27b`) and seamlessly falls back to a lightweight dev model (`qwen2.5-coder:7b`) if necessary, ensuring zero configuration friction across different environments.
- **Defensive Extraction & Formatting**: Implements robust regex parsing pipelines to strip conversational filler, code blocks, and markdown wrappers, guaranteeing that the `output.json` adheres strictly to the required schema.
- **Fault-Tolerant Loop**: Wrapped in comprehensive exception and timeout handlers. If a prompt stalls or errors out, it gracefully outputs `"ERROR"` per problem ID rather than crashing the entire batch evaluation job.

---

## 📦 Project Structure
```text
├── main.py            # Core agent logic, model detection & CoT pipeline
├── requirements.txt   # Minimalist dependencies (ollama)
└── README.md          # Documentation & setup guide