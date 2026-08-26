import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_site
import generate_summaries


class BrandingTests(unittest.TestCase):
    def test_default_homepage_is_autonomous_driving(self):
        with patch.dict(os.environ, {}, clear=True):
            html = build_site.generate_index_html()

        self.assertIn("自动驾驶 每日论文卡", html)
        self.assertIn("Autonomous Driving Research Feed", html)
        self.assertIn("UniAD、BEVFormer、nuScenes", html)
        self.assertNotIn("VLA/WAM", html)
        self.assertNotIn("World Action Model Feed", html)

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


if __name__ == "__main__":
    unittest.main()
