import os

import yaml


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


def load_yaml_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_cli_defaults(config):
    runtime = config.get("runtime", {})
    llm = config.get("llm", {})
    email = config.get("email", {})
    output = config.get("output", {})

    defaults = {
        "categories": config.get("categories"),
        "max_entries": runtime.get("max_entries"),
        "max_paper_num": runtime.get("max_paper_num"),
        "provider": llm.get("provider"),
        "model": llm.get("model"),
        "base_url": llm.get("base_url"),
        "temperature": runtime.get("temperature"),
        "num_workers": runtime.get("num_workers"),
        "title": runtime.get("title"),
        "save": runtime.get("save"),
        "save_dir": output.get("save_dir"),
        "report_dir": output.get("report_dir"),
        "no_email": output.get("no_email"),
        "smtp_server": email.get("smtp_server"),
        "smtp_port": email.get("smtp_port"),
        "sender": email.get("sender"),
        "receiver": email.get("receiver"),
        "sender_password": email.get("sender_password"),
    }
    return {key: value for key, value in defaults.items() if value is not None}
