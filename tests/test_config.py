import os
import tempfile
import unittest
from pathlib import Path

from util.config import build_cli_defaults, load_env_file, load_yaml_config, resolve_llm_config


class ConfigTest(unittest.TestCase):
    def test_deepseek_config_loads_api_key_from_env_file(self):
        old_deepseek_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        old_openai_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                env_path = Path(tmpdir) / ".env"
                env_path.write_text("DEEPSEEK_API_KEY=deepseek-test-key\n")

                load_env_file(str(env_path))
                provider, model, base_url, api_key = resolve_llm_config("DeepSeek")

                self.assertEqual(provider, "DeepSeek")
                self.assertEqual(model, "deepseek-chat")
                self.assertEqual(base_url, "https://api.deepseek.com")
                self.assertEqual(api_key, "deepseek-test-key")
        finally:
            os.environ.pop("DEEPSEEK_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            if old_deepseek_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
            if old_openai_key is not None:
                os.environ["OPENAI_API_KEY"] = old_openai_key

    def test_openai_api_key_fallback(self):
        old_deepseek_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        old_openai_key = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = "openai-fallback-key"

            provider, model, base_url, api_key = resolve_llm_config("DeepSeek")

            self.assertEqual(provider, "DeepSeek")
            self.assertEqual(model, "deepseek-chat")
            self.assertEqual(base_url, "https://api.deepseek.com")
            self.assertEqual(api_key, "openai-fallback-key")
        finally:
            os.environ.pop("DEEPSEEK_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            if old_deepseek_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
            if old_openai_key is not None:
                os.environ["OPENAI_API_KEY"] = old_openai_key

    def test_load_yaml_config_builds_cli_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
categories:
  - quant-ph
description: |
  quantum metrology
runtime:
  max_entries: 5
  max_paper_num: 12
  num_workers: 2
  temperature: 0.2
  title: Quantum Daily
  save: true
output:
  save_dir: output
  report_dir: daily_reports
  no_email: true
llm:
  provider: DeepSeek
  model: deepseek-chat
  base_url: https://api.deepseek.com
email:
  receiver: 2824334214@qq.com
""",
                encoding="utf-8",
            )

            config = load_yaml_config(str(config_path))
            defaults = build_cli_defaults(config)

            self.assertEqual(config["description"].strip(), "quantum metrology")
            self.assertEqual(defaults["categories"], ["quant-ph"])
            self.assertEqual(defaults["provider"], "DeepSeek")
            self.assertEqual(defaults["model"], "deepseek-chat")
            self.assertEqual(defaults["base_url"], "https://api.deepseek.com")
            self.assertEqual(defaults["max_entries"], 5)
            self.assertEqual(defaults["max_paper_num"], 12)
            self.assertEqual(defaults["num_workers"], 2)
            self.assertEqual(defaults["temperature"], 0.2)
            self.assertEqual(defaults["title"], "Quantum Daily")
            self.assertTrue(defaults["save"])
            self.assertEqual(defaults["save_dir"], "output")
            self.assertEqual(defaults["report_dir"], "daily_reports")
            self.assertTrue(defaults["no_email"])
            self.assertEqual(defaults["receiver"], "2824334214@qq.com")


if __name__ == "__main__":
    unittest.main()
