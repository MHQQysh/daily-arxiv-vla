import os
import sys
import tempfile
import threading
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

        with patch.dict(
            os.environ,
            {
                "API_MAX_RETRIES": "1",
                # 摘要任务必须显式禁用 thinking，不能被同名环境变量重新开启。
                "DEEPSEEK_THINKING": "enabled",
            },
            clear=True,
        ):
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
        self.assertEqual(request["max_tokens"], 2048)
        self.assertEqual(
            request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
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

    def test_workflow_allows_one_hundred_summaries_with_five_workers(self):
        workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("SUMMARY_MAX_ITEMS: 100", workflow)
        self.assertIn("SUMMARY_WORKERS: 5", workflow)

    def test_summary_item_limit_is_one_hundred_and_workers_are_capped_at_five(self):
        with patch.dict(os.environ, {"SUMMARY_MAX_ITEMS": "100"}, clear=True):
            self.assertEqual(generate_summaries.get_summary_item_limit(), 100)

        with patch.dict(os.environ, {"SUMMARY_WORKERS": "99"}, clear=True):
            self.assertEqual(generate_summaries.get_summary_workers(8), 5)
            self.assertEqual(generate_summaries.get_summary_workers(3), 3)

        with patch.dict(os.environ, {"SUMMARY_WORKERS": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SUMMARY_WORKERS"):
                generate_summaries.get_summary_workers(5)

        with patch.dict(os.environ, {"SUMMARY_MAX_ITEMS": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SUMMARY_MAX_ITEMS"):
                generate_summaries.get_summary_item_limit()


class SummaryBatchTests(unittest.TestCase):
    HEADER = (
        "| 日期 | 标题 | 链接 | 简要总结 |\n"
        "| --- | --- | --- | --- |\n"
    )

    @staticmethod
    def pending_line(index: int) -> str:
        return (
            f"| 2026-08-{index + 1:02d} | Paper {index} | "
            f"https://arxiv.org/abs/2608.{index:05d} | "
            "<details><summary>展开</summary>待生成</details> |\n"
        )

    def test_each_run_processes_only_first_five_pending_entries_concurrently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_path = Path(temp_dir) / "papers.md"
            completed_line = (
                "| 2026-08-20 | Completed | https://arxiv.org/abs/2608.99999 | "
                "<details><summary>展开</summary>已完成</details> |\n"
            )
            papers_path.write_text(
                self.HEADER
                + "".join(self.pending_line(index) for index in range(7))
                + completed_line,
                encoding="utf-8",
            )

            lock = threading.Lock()
            release = threading.Event()
            active_count = 0
            started_count = 0
            peak_active = 0

            def summarize(_client, link):
                nonlocal active_count, started_count, peak_active
                with lock:
                    active_count += 1
                    started_count += 1
                    peak_active = max(peak_active, active_count)
                    if started_count >= 3:
                        release.set()
                if not release.wait(timeout=2):
                    raise RuntimeError("并发任务没有同时启动")
                with lock:
                    active_count -= 1
                return f"## 论文概述<br>- 已生成 {link}"

            client = object()
            env = {
                "SUMMARY_WORKERS": "3",
                "SUMMARY_MAX_ITEMS": "5",
                "BATCH_WRITE_SIZE": "5",
            }
            with patch.dict(os.environ, env, clear=True):
                with patch.object(
                    generate_summaries,
                    "get_papers_md_path",
                    return_value=str(papers_path),
                ):
                    with patch.object(
                        generate_summaries,
                        "get_client",
                        return_value=client,
                    ) as get_client:
                        with patch.object(
                            generate_summaries,
                            "generate_summary_for_link",
                            side_effect=summarize,
                        ) as generate_summary:
                            with patch("builtins.print"):
                                selected, updated = generate_summaries.update_papers_md()

            self.assertEqual((selected, updated), (5, 5))
            get_client.assert_called_once_with()
            self.assertEqual(generate_summary.call_count, 5)
            requested_links = {
                call.args[1] for call in generate_summary.call_args_list
            }
            self.assertEqual(
                requested_links,
                {
                    f"https://arxiv.org/abs/2608.{index:05d}"
                    for index in range(5)
                },
            )
            self.assertEqual(peak_active, 3)

            body_lines = papers_path.read_text(encoding="utf-8").splitlines()[2:]
            for line in body_lines[:5]:
                self.assertNotIn("待生成", line)
            for line in body_lines[5:7]:
                self.assertIn("待生成", line)
            self.assertIn("已完成", body_lines[7])

    def test_failed_future_keeps_placeholder_and_other_results_are_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_path = Path(temp_dir) / "papers.md"
            papers_path.write_text(
                self.HEADER
                + "".join(self.pending_line(index) for index in range(3)),
                encoding="utf-8",
            )

            def summarize(_client, link):
                if link.endswith("00001"):
                    raise RuntimeError("mock failure")
                return "## 论文概述<br>- 成功"

            with patch.dict(os.environ, {"SUMMARY_WORKERS": "2"}, clear=True):
                with patch.object(
                    generate_summaries,
                    "get_papers_md_path",
                    return_value=str(papers_path),
                ):
                    with patch.object(generate_summaries, "get_client", return_value=object()):
                        with patch.object(
                            generate_summaries,
                            "generate_summary_for_link",
                            side_effect=summarize,
                        ):
                            with patch("builtins.print") as printer:
                                selected, updated = generate_summaries.update_papers_md()

            self.assertEqual((selected, updated), (3, 2))
            body_lines = papers_path.read_text(encoding="utf-8").splitlines()[2:]
            self.assertNotIn("待生成", body_lines[0])
            self.assertIn("待生成", body_lines[1])
            self.assertNotIn("待生成", body_lines[2])
            rendered_log = "\n".join(
                " ".join(str(part) for part in call.args)
                for call in printer.call_args_list
            )
            self.assertIn("https://arxiv.org/abs/2608.00001", rendered_log)

    def test_no_pending_entries_avoids_creating_an_api_client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_path = Path(temp_dir) / "papers.md"
            papers_path.write_text(
                self.HEADER
                + "| 2026-08-20 | Completed | https://arxiv.org/abs/2608.99999 | "
                "<details><summary>展开</summary>已完成</details> |\n",
                encoding="utf-8",
            )

            with patch.object(
                generate_summaries,
                "get_papers_md_path",
                return_value=str(papers_path),
            ):
                with patch.object(generate_summaries, "get_client") as get_client:
                    selected, updated = generate_summaries.update_papers_md()

            self.assertEqual((selected, updated), (0, 0))
            get_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
