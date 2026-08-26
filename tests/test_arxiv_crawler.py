import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from arxiv_crawler import ArxivCollector
from autonomous_driving_topics import TOPIC_QUERIES, is_relevant_paper


def paper(arxiv_id, title, day, category="cs.CV", summary="autonomous driving perception"):
    return SimpleNamespace(
        entry_id=f"https://arxiv.org/abs/{arxiv_id}",
        title=title,
        summary=summary,
        primary_category=category,
        published=datetime(2026, 8, day, tzinfo=timezone.utc),
    )


class TopicRegistryTests(unittest.TestCase):
    def test_has_seven_named_topics(self):
        self.assertEqual(
            list(TOPIC_QUERIES),
            ["overview", "perception", "localization_mapping", "prediction", "planning_decision", "control", "end_to_end_foundation"],
        )

    def test_relevance_requires_driving_context(self):
        self.assertTrue(is_relevant_paper("End-to-End Autonomous Driving", "camera policy"))
        self.assertTrue(is_relevant_paper("BEV Perception for Road Vehicles", "3D detection"))
        self.assertFalse(is_relevant_paper("Generic 3D Object Detection", "indoor point clouds"))


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.papers_path = str(Path(self.temp_dir.name) / "papers.md")

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_collector(self):
        return ArxivCollector(
            self.papers_path,
            init_results=2,
            daily_results=1,
            topic_queries={"one": "q1", "two": "q2"},
        )

    def test_collect_deduplicates_versions_and_sorts_descending(self):
        collector = self.make_collector()
        first = paper("2608.00001v1", "Autonomous Driving A", 20)
        duplicate = paper("2608.00001v2", "Autonomous Driving A revised", 21)
        newest = paper("2608.00002v1", "Autonomous Driving B", 22)
        with patch.object(collector, "_search", side_effect=[[first, newest], [duplicate]]):
            results = collector._collect(2)
        self.assertEqual([collector._normalize_link(item.entry_id) for item in results], [
            "https://arxiv.org/abs/2608.00002",
            "https://arxiv.org/abs/2608.00001",
        ])

    def test_collect_continues_after_one_topic_fails(self):
        collector = self.make_collector()
        valid = paper("2608.00003v1", "Planning for Autonomous Vehicles", 23)
        with patch.object(collector, "_search", side_effect=[RuntimeError("rate limit"), [valid]]):
            self.assertEqual(collector._collect(1), [valid])

    def test_collect_raises_when_all_topics_fail(self):
        collector = self.make_collector()
        with patch.object(collector, "_search", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "全部 2 个自动驾驶主题检索失败"):
                collector._collect(1)

    def test_header_only_file_is_an_empty_library(self):
        collector = self.make_collector()
        Path(self.papers_path).write_text(
            "| 日期 | 标题 | 链接 | 简要总结 |\n| --- | --- | --- | --- |\n",
            encoding="utf-8",
        )
        self.assertFalse(collector.has_existing_papers())


if __name__ == "__main__":
    unittest.main()
