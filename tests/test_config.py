import os
import tempfile
import unittest
from pathlib import Path

from util.config import load_env_file, resolve_llm_config


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


if __name__ == "__main__":
    unittest.main()
