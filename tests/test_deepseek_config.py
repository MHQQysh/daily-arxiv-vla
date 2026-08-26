import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_summaries
import test_api


class DeepSeekConfigTests(unittest.TestCase):
    def test_defaults_are_official_flash(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            with patch.object(generate_summaries, "load_dotenv"):
                with patch.object(generate_summaries, "OpenAI") as openai_class:
                    client = generate_summaries.get_client()

            self.assertEqual(generate_summaries.get_model(), "deepseek-v4-flash")

        self.assertIs(client, openai_class.return_value)
        openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )

    def test_missing_deepseek_key_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(generate_summaries, "load_dotenv"):
                with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                    generate_summaries.get_client()

    def test_deepseek_endpoint_and_model_can_be_overridden(self):
        env = {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://gateway.example/deepseek ",
            "DEEPSEEK_MODEL": "deepseek-v4-flash-custom ",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(generate_summaries, "load_dotenv"):
                with patch.object(generate_summaries, "OpenAI") as openai_class:
                    generate_summaries.get_client()
            model = generate_summaries.get_model()

        openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://gateway.example/deepseek",
        )
        self.assertEqual(model, "deepseek-v4-flash-custom")

    def test_summary_request_selects_default_flash_model(self):
        http_response = MagicMock()
        http_response.text = "<html><body>autonomous driving paper</body></html>"
        api_response = MagicMock()
        api_response.choices = [MagicMock()]
        api_response.choices[0].message.content = "## 论文概述\n- 模拟摘要"
        client = MagicMock()
        client.chat.completions.create.return_value = api_response

        with patch.dict(os.environ, {"API_MAX_RETRIES": "1"}, clear=True):
            with patch.object(
                generate_summaries.requests,
                "get",
                return_value=http_response,
            ) as http_get:
                with patch("builtins.print"):
                    result = generate_summaries.generate_summary_for_link(
                        client,
                        "https://arxiv.org/abs/2608.00001",
                    )

        http_get.assert_called_once_with(
            "https://arxiv.org/html/2608.00001",
            timeout=30,
        )
        http_response.raise_for_status.assert_called_once_with()
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "deepseek-v4-flash")
        self.assertFalse(request["stream"])
        self.assertIs(
            request["messages"][0]["content"],
            generate_summaries.AUTONOMOUS_DRIVING_SUMMARY_PROMPT,
        )
        self.assertEqual(result, "## 论文概述<br>- 模拟摘要")

    def test_api_retries_only_the_selected_deepseek_model(self):
        http_response = MagicMock()
        http_response.text = "<html>paper</html>"
        api_response = MagicMock()
        api_response.choices = [MagicMock()]
        api_response.choices[0].message.content = "## 核心贡献\n- 成功"
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("temporary failure"),
            api_response,
        ]

        with patch.dict(os.environ, {"API_MAX_RETRIES": "2"}, clear=True):
            with patch.object(generate_summaries.requests, "get", return_value=http_response):
                with patch.object(generate_summaries.time, "sleep") as sleep:
                    with patch("builtins.print"):
                        result = generate_summaries.generate_summary_for_link(
                            client,
                            "https://arxiv.org/abs/2608.00002",
                            model="deepseek-v4-flash",
                        )

        self.assertEqual(client.chat.completions.create.call_count, 2)
        requested_models = [
            call.kwargs["model"]
            for call in client.chat.completions.create.call_args_list
        ]
        self.assertEqual(requested_models, ["deepseek-v4-flash", "deepseek-v4-flash"])
        sleep.assert_called_once_with(1)
        self.assertEqual(result, "## 核心贡献<br>- 成功")

    def test_manual_probe_rejects_missing_key_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(test_api, "load_dotenv"):
                with patch.object(test_api, "OpenAI") as openai_class:
                    with patch.object(test_api.requests, "get") as http_get:
                        with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                            test_api.main()

        openai_class.assert_not_called()
        http_get.assert_not_called()

    def test_api_error_log_redacts_configured_key(self):
        fake_secret = "unit-test-secret-value"
        http_response = MagicMock()
        http_response.text = "<html>paper</html>"
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError(
            f"upstream error included {fake_secret}"
        )
        env = {
            "API_MAX_RETRIES": "1",
            "DEEPSEEK_API_KEY": fake_secret,
        }

        with patch.dict(os.environ, env, clear=True):
            with patch.object(generate_summaries.requests, "get", return_value=http_response):
                with patch("builtins.print") as printer:
                    result = generate_summaries.generate_summary_for_link(
                        client,
                        "https://arxiv.org/abs/2608.00003",
                    )

        rendered_log = "\n".join(
            " ".join(str(part) for part in call.args)
            for call in printer.call_args_list
        )
        self.assertEqual(result, "")
        self.assertNotIn(fake_secret, rendered_log)
        self.assertIn("[REDACTED]", rendered_log)

    def test_no_legacy_provider_configuration_remains(self):
        for relative_path in ("scripts/generate_summaries.py", "test_api.py"):
            with self.subTest(path=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("MODELSCOPE", source.upper())
                self.assertNotIn("api-inference.modelscope.cn", source)


if __name__ == "__main__":
    unittest.main()
