from arxiv_daily import ArxivDaily
from util.config import (
    build_cli_defaults,
    load_env_file,
    load_yaml_config,
    resolve_llm_config,
)
import argparse
import os
import sys

if __name__ == "__main__":
    load_env_file()

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_yaml_config(config_args.config)
    config_defaults = build_cli_defaults(config)

    parser = argparse.ArgumentParser(description="Arxiv Daily")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--categories",
        nargs="+",
        help="categories",
        default=config_defaults.get("categories"),
    )
    parser.add_argument(
        "--max_paper_num",
        type=int,
        help="max_paper_num",
        default=config_defaults.get("max_paper_num", 60),
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        help="max_entries to get from arxiv",
        default=config_defaults.get("max_entries", 100),
    )
    parser.add_argument(
        "--provider", type=str, help="provider", default=config_defaults.get("provider")
    )
    parser.add_argument(
        "--model", type=str, help="model", default=config_defaults.get("model")
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=config_defaults.get("save", False),
        help="Save the email content to a file.",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="Disable saving when enabled by config.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=config_defaults.get("save_dir", "./arxiv_history"),
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=config_defaults.get("report_dir"),
        help="Directory for Markdown reports. If set, reports are saved as YYYY-MM-DD.md in this directory.",
    )

    parser.add_argument(
        "--base_url", type=str, help="base_url", default=config_defaults.get("base_url")
    )
    parser.add_argument("--api_key", type=str, help="api_key", default=None)

    parser.add_argument(
        "--description",
        type=str,
        help="Path to the file that describes your interested research area.",
        default="description.txt",
    )

    parser.add_argument(
        "--smtp_server",
        type=str,
        help="SMTP server",
        default=config_defaults.get("smtp_server"),
    )
    parser.add_argument(
        "--smtp_port",
        type=int,
        help="SMTP port",
        default=config_defaults.get("smtp_port"),
    )
    parser.add_argument(
        "--sender",
        type=str,
        help="Sender email address",
        default=config_defaults.get("sender"),
    )
    parser.add_argument(
        "--receiver",
        type=str,
        help="Receiver email address",
        default=config_defaults.get("receiver"),
    )
    parser.add_argument(
        "--sender_password",
        type=str,
        help="Sender email password",
        default=config_defaults.get("sender_password"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature",
        default=config_defaults.get("temperature", 0.7),
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers",
        default=config_defaults.get("num_workers", 4),
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title of the email",
        default=config_defaults.get("title", "Daily arXiv"),
    )
    parser.add_argument(
        "--include-seen",
        action="store_true",
        help="Reprocess papers that already exist in the SQLite history.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        default=config_defaults.get("no_email", False),
        help="Generate reports without sending email.",
    )
    parser.add_argument(
        "--email",
        action="store_false",
        dest="no_email",
        help="Enable email sending when disabled by config.",
    )

    args = parser.parse_args()
    if args.categories is None:
        parser.error("the following arguments are required: --categories")
    if args.provider is None:
        parser.error("the following arguments are required: --provider")

    args.provider, args.model, args.base_url, args.api_key = resolve_llm_config(
        args.provider,
        args.model,
        args.base_url,
        args.api_key,
    )

    if not (args.provider == "Ollama" or args.provider == "ollama"):
        assert args.base_url is not None, (
            "base_url is required for OpenAI-compatible providers"
        )
        assert args.api_key is not None, (
            "api_key is required. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or --api_key."
        )

    if config_args.config and "--description" not in sys.argv and config.get("description"):
        args.description = config["description"]
    else:
        with open(args.description, "r") as f:
            args.description = f.read()

    # Test LLM availability
    if args.provider == "Ollama" or args.provider == "ollama":
        from llm.Ollama import Ollama

        try:
            model = Ollama(args.model)
            model.inference("Hello, who are you?")
        except Exception as e:
            print(e)
            assert False, "Model not initialized successfully."
    elif (
        args.provider == "OpenAI"
        or args.provider == "openai"
        or args.provider == "SiliconFlow"
        or args.provider == "DeepSeek"
        or args.provider == "deepseek"
    ):
        from llm.GPT import GPT

        try:
            model = GPT(args.model, args.base_url, args.api_key)
            model.inference("Hello, who are you?")
        except Exception as e:
            print(e)
            assert False, "Model not initialized successfully."
    else:
        assert False, "Model not supported."

    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = None

    arxiv_daily = ArxivDaily(
        args.categories,
        args.max_entries,
        args.max_paper_num,
        args.provider,
        args.model,
        args.base_url,
        args.api_key,
        args.description,
        args.num_workers,
        args.temperature,
        args.save_dir,
        args.include_seen,
        report_dir=args.report_dir,
    )

    if args.no_email:
        arxiv_daily.get_recommendation()
    else:
        arxiv_daily.send_email(
            args.sender,
            args.receiver,
            args.sender_password,
            args.smtp_server,
            args.smtp_port,
            args.title,
        )
