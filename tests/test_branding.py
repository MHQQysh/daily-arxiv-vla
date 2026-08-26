import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_site
import generate_summaries


class BrandingTests(unittest.TestCase):
    @staticmethod
    def make_built_record():
        return build_site.build_site_records(
            [
                {
                    "date": "2026-08-26",
                    "title": "Safe Autonomous Driving Paper",
                    "link": "https://arxiv.org/abs/2608.00001",
                    "details_raw": "## 核心贡献\n- 验证页面品牌与安全渲染。",
                }
            ],
            {},
        )[0]

    def test_default_homepage_is_autonomous_driving(self):
        with patch.dict(os.environ, {}, clear=True):
            html = build_site.generate_index_html()

        self.assertIn("自动驾驶每日论文卡", html)
        self.assertNotIn("自动驾驶 每日论文卡", html)
        self.assertIn(
            '<h1 class="site-title">自动驾驶<span class="site-title-nowrap">每日论文卡</span></h1>',
            html,
        )
        self.assertIn("Autonomous Driving Research Feed", html)
        self.assertIn("UniAD、BEVFormer、nuScenes", html)
        self.assertNotIn("VLA/WAM", html)
        self.assertNotIn("World Action Model Feed", html)

    def test_detail_and_cover_use_the_exact_default_site_title(self):
        record = self.make_built_record()
        with patch.dict(os.environ, {}, clear=True):
            detail_html = build_site.generate_paper_html(record)
            cover_html = build_site.generate_cover_html(record)

        expected = '<span class="detail-site-name">自动驾驶每日论文卡</span>'
        self.assertIn(expected, detail_html)
        self.assertIn(expected, cover_html)
        self.assertIn(
            "<title>Safe Autonomous Driving Paper - 自动驾驶每日论文卡</title>",
            detail_html,
        )
        self.assertNotIn("自动驾驶 每日论文卡", detail_html)
        self.assertNotIn("自动驾驶 每日论文卡", cover_html)

    def test_keyword_label_has_safe_display_override(self):
        with patch.dict(
            os.environ,
            {
                "ARXIV_KEYWORD_LABEL": "智能驾驶",
                "ARXIV_QUERY_KEYWORD": 'all:"untrusted query syntax"',
            },
            clear=True,
        ):
            self.assertEqual(build_site.get_arxiv_keyword_label(), "智能驾驶")
            html = build_site.generate_index_html()

        self.assertIn("智能驾驶", html)
        self.assertNotIn("untrusted query syntax", html)

    def test_keyword_label_falls_back_when_blank_and_is_html_escaped(self):
        with patch.dict(os.environ, {"ARXIV_KEYWORD_LABEL": "   "}, clear=True):
            self.assertEqual(build_site.get_arxiv_keyword_label(), "自动驾驶")

        with patch.dict(
            os.environ,
            {"ARXIV_KEYWORD_LABEL": '<script>alert("unsafe")</script>'},
            clear=True,
        ):
            html = build_site.generate_index_html()

        self.assertNotIn('<script>alert("unsafe")</script>', html)
        self.assertIn("&lt;script&gt;alert", html)

    def test_summary_prompt_requests_driving_evidence(self):
        prompt = generate_summaries.AUTONOMOUS_DRIVING_SUMMARY_PROMPT
        for phrase in (
            "自动驾驶",
            "传感器",
            "数据集",
            "评估指标",
            "开环",
            "闭环",
            "不得补造",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)

    def test_summary_generation_uses_the_autonomous_driving_system_prompt(self):
        http_response = MagicMock()
        http_response.text = "<html><body>paper source</body></html>"
        api_response = MagicMock()
        api_response.choices = [MagicMock()]
        api_response.choices[0].message.content = "## 论文概述\n- 测试摘要"
        client = MagicMock()
        client.chat.completions.create.return_value = api_response

        with patch.object(generate_summaries, "RATE_LIMITED_MODELS", set()):
            with patch.object(
                generate_summaries.requests,
                "get",
                return_value=http_response,
            ) as http_get:
                with patch("builtins.print"):
                    result = generate_summaries.generate_summary_for_link(
                        client,
                        "https://arxiv.org/abs/2608.00001",
                        model="mock-model",
                    )

        http_get.assert_called_once_with("https://arxiv.org/html/2608.00001", timeout=30)
        client.chat.completions.create.assert_called_once()
        call_kwargs = client.chat.completions.create.call_args.kwargs
        self.assertIs(
            call_kwargs["messages"][0]["content"],
            generate_summaries.AUTONOMOUS_DRIVING_SUMMARY_PROMPT,
        )
        self.assertEqual(result, "## 论文概述<br>- 测试摘要")


class MarkdownSafetyTests(unittest.TestCase):
    def test_raw_html_and_event_handlers_are_escaped(self):
        rendered = build_site.markdown_to_html(
            '<script>alert("xss")</script>\n'
            '- <img src=x onerror="alert(1)"> **保留粗体** `保留代码`'
        )

        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img ", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", rendered)
        self.assertIn("<strong>保留粗体</strong>", rendered)
        self.assertIn("<code>保留代码</code>", rendered)
        self.assertIn("<ul><li>", rendered)

    def test_unsafe_markdown_links_become_plain_escaped_labels(self):
        rendered = build_site.markdown_to_html(
            "[脚本](javascript:evil) "
            "[数据](data:text/html;base64,PHNjcmlwdD4=) "
            "[空主机](https://@) "
            "[仅端口](https://:443/path) "
            "[坏端口](https://example.com:bad/path)"
        )

        self.assertNotIn("<a ", rendered)
        self.assertNotIn("javascript:", rendered)
        self.assertNotIn("data:text/html", rendered)
        self.assertIn("脚本", rendered)
        self.assertIn("数据", rendered)
        self.assertIn("空主机", rendered)
        self.assertIn("仅端口", rendered)
        self.assertIn("坏端口", rendered)

    def test_inline_code_inside_link_destination_does_not_corrupt_href(self):
        rendered = build_site.markdown_to_html(
            "[损坏链接](https://example.com/`fragment`)"
        )

        self.assertNotIn("<a ", rendered)
        self.assertNotIn("<code>", rendered)
        self.assertIn("损坏链接", rendered)

    def test_https_markdown_link_is_preserved_with_quote_safe_href(self):
        rendered = build_site.markdown_to_html(
            '[论文](https://example.com/paper?q="x"&lang=zh)'
        )

        self.assertIn(
            '<a href="https://example.com/paper?q=&quot;x&quot;&amp;lang=zh" '
            'target="_blank" rel="noopener noreferrer">论文</a>',
            rendered,
        )

    def test_html_inside_supported_inline_markdown_remains_escaped(self):
        rendered = build_site.markdown_to_html(
            "**<span onclick=evil>粗体</span>** "
            "`<img src=x onerror=evil>` "
            "[<b>链接</b>](https://example.com)"
        )

        self.assertNotIn("<span onclick", rendered)
        self.assertNotIn("<img src", rendered)
        self.assertNotIn("<b>", rendered)
        self.assertIn("<strong>&lt;span onclick=evil&gt;粗体&lt;/span&gt;</strong>", rendered)
        self.assertIn("<code>&lt;img src=x onerror=evil&gt;</code>", rendered)
        self.assertIn("&lt;b&gt;链接&lt;/b&gt;</a>", rendered)


if __name__ == "__main__":
    unittest.main()
