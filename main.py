from arxiv_daily import ArxivDaily
from util.config import load_env_file, resolve_llm_config
import argparse
import os

if __name__ == "__main__":
    load_env_file()

    parser = argparse.ArgumentParser(description="Arxiv Daily")
    parser.add_argument("--categories", nargs="+", help="categories", required=True)
    parser.add_argument("--max_paper_num", type=int, help="max_paper_num", default=60)
    parser.add_argument(
        "--max_entries", type=int, help="max_entries to get from arxiv", default=100
    )
    parser.add_argument("--provider", type=str, help="provider", required=True)
    parser.add_argument("--model", type=str, help="model", required=None)
    parser.add_argument(
        "--save", action="store_true", help="Save the email content to a file."
    )
    parser.add_argument("--save_dir", type=str, default="./arxiv_history")
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Directory for Markdown reports. If set, reports are saved as YYYY-MM-DD.md in this directory.",
    )

    parser.add_argument("--base_url", type=str, help="base_url", default=None)
    parser.add_argument("--api_key", type=str, help="api_key", default=None)

    parser.add_argument(
        "--description",
        type=str,
        help="Path to the file that describes your interested research area.",
        default="description.txt",
    )

    parser.add_argument("--smtp_server", type=str, help="SMTP server")
    parser.add_argument("--smtp_port", type=int, help="SMTP port")
    parser.add_argument("--sender", type=str, help="Sender email address")
    parser.add_argument("--receiver", type=str, help="Receiver email address")
    parser.add_argument("--sender_password", type=str, help="Sender email password")
    parser.add_argument("--temperature", type=float, help="Temperature", default=0.7)

    parser.add_argument("--num_workers", type=int, help="Number of workers", default=4)
    parser.add_argument(
        "--title", type=str, help="Title of the email", default="Daily arXiv"
    )
    parser.add_argument(
        "--include-seen",
        action="store_true",
        help="Reprocess papers that already exist in the SQLite history.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Generate reports without sending email.",
    )

    args = parser.parse_args()

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
