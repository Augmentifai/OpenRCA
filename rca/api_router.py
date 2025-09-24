import os
import yaml
import time
from litellm import completion

def load_config(config_path="rca/api_config.yaml"):
    """
    Load environment variables and YAML config into a single dictionary.
    """
    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs

configs = load_config()


def LiteLLM_chat_completion(messages, temperature=0.0):
    """
    Universal chat completion through LiteLLM.
    Supports OpenAI, Anthropic, Google, Azure, local Ollama, and many others.
    """
    # LiteLLM expects messages in the same structure as OpenAI:
    # [{"role": "system"|"user"|"assistant", "content": "..."}]
    response = completion(
        model=configs["MODEL"],            # e.g., "gpt-4o", "claude-3-opus", "gemini-pro", "ollama/llama2"
        messages=messages,
        temperature=temperature,
        api_base=configs.get("API_BASE"),  # Optional: required for custom endpoints (like Ollama)
        api_key=configs.get("API_KEY")     # For local Ollama you can leave API_KEY unset
    )
    # LiteLLM returns a response object similar to OpenAI
    return response["choices"][0]["message"]["content"]


def get_chat_completion(messages, temperature=0.0):
    """
    Unified entry point for chat completion.
    Retries up to 3 times on 429 (rate limit) errors.
    """
    def send_request():
        return LiteLLM_chat_completion(messages, temperature)

    for i in range(3):
        try:
            return send_request()
        except Exception as e:
            print(e)
            if '429' in str(e):
                print("Rate limit exceeded. Waiting for 1 second.")
                time.sleep(1)
                continue
            else:
                raise e