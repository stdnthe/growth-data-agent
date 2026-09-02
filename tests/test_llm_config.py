from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.llm_sql import resolve_llm_config, validate_compatible_base_url


class LLMConfigTests(unittest.TestCase):
    def test_deepseek_remains_the_default_provider(self) -> None:
        env = {
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_MODEL": "deepseek-chat",
        }
        with patch.dict(os.environ, env, clear=True):
            config = resolve_llm_config()
        self.assertEqual(config["provider"], "deepseek")
        self.assertEqual(config["model"], "deepseek-chat")
        self.assertEqual(config["api_key"], "deepseek-key")

    def test_openai_provider_uses_its_own_environment(self) -> None:
        env = {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_MODEL": "gpt-test",
        }
        with patch.dict(os.environ, env, clear=True):
            config = resolve_llm_config()
        self.assertEqual(config["provider"], "openai")
        self.assertEqual(config["model"], "gpt-test")
        self.assertEqual(config["api_key"], "openai-key")
        self.assertEqual(config["base_url"], "https://api.openai.com/v1")

    def test_openai_compatible_accepts_explicit_runtime_values(self) -> None:
        config = resolve_llm_config(
            provider="openai-compatible",
            model="custom-model",
            api_key_override="custom-key",
            base_url_override="https://models.example.com/v1",
        )
        self.assertEqual(config["provider"], "openai_compatible")
        self.assertEqual(config["model"], "custom-model")
        self.assertEqual(config["api_key"], "custom-key")
        self.assertEqual(config["base_url"], "https://models.example.com/v1")

    def test_request_api_key_override_takes_precedence_without_being_logged(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "server-key"}, clear=False):
            config = resolve_llm_config(api_key_override=" visitor-key ")
        self.assertEqual(config["api_key"], "visitor-key")
        self.assertEqual(config["api_key_source"], "request")

    def test_public_ui_can_disable_environment_key_fallback(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "server-key"}, clear=False):
            config = resolve_llm_config(provider="openai", allow_stored_api_key=False)
        self.assertIsNone(config["api_key"])
        self.assertEqual(config["api_key_source"], "missing")

    def test_custom_base_url_rejects_local_and_non_https_targets(self) -> None:
        self.assertFalse(validate_compatible_base_url("http://models.example.com/v1")[0])
        self.assertFalse(validate_compatible_base_url("https://127.0.0.1:9000/v1")[0])
        self.assertFalse(validate_compatible_base_url("https://service.internal/v1")[0])
        self.assertTrue(validate_compatible_base_url("https://models.example.com/v1")[0])

    def test_unknown_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            resolve_llm_config(provider="unknown")


if __name__ == "__main__":
    unittest.main()
