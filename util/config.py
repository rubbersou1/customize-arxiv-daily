import os


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return {}

    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
            values[key] = value
    return values


def get_api_key(cli_api_key=None):
    return (
        cli_api_key
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def resolve_llm_config(provider, model=None, base_url=None, api_key=None):
    provider_name = provider or "DeepSeek"
    if provider_name.lower() == "deepseek":
        model = model or "deepseek-chat"
        base_url = base_url or "https://api.deepseek.com"
    return provider_name, model, base_url, get_api_key(api_key)
