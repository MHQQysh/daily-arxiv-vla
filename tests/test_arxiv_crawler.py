import os
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


def paper(
    arxiv_id,
    title,
    day,
    category="cs.CV",
    summary="autonomous driving perception",
    updated_day=None,
    scheme="https",
):
    return SimpleNamespace(
        entry_id=f"{scheme}://arxiv.org/abs/{arxiv_id}",
        title=title,
        summary=summary,
        primary_category=category,
        published=datetime(2026, 8, day, tzinfo=timezone.utc),
        updated=datetime(2026, 8, updated_day or day, tzinfo=timezone.utc),
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

    def test_relevance_uses_word_boundaries(self):
        adversarial_papers = (
            ("Broad Object Detection", "indoor benchmarks"),
            ("Planet Formation", "road traffic analysis"),
            ("Islamic History", "road traffic archives"),
            ("Beverage Recognition", "road traffic imagery"),
        )
        for title, summary in adversarial_papers:
            with self.subTest(title=title):
                self.assertFalse(is_relevant_paper(title, summary))

    def test_prediction_query_and_relevance_cover_common_aliases(self):
        prediction_query = TOPIC_QUERIES["prediction"]
        self.assertIn('all:"trajectory forecasting"', prediction_query)
        self.assertIn('all:"motion prediction"', prediction_query)
        self.assertTrue(is_relevant_paper(
            "Heterogeneous Trajectory Forecasting via Risk and Scene Graph Learning",
            "Trajectory forecasting for heterogeneous road agents in driving scenes.",
        ))
        self.assertTrue(is_relevant_paper("Motion Prediction", "automotive scenes"))


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
        del first.updated
        del duplicate.updated
        with patch.object(collector, "_search", side_effect=[[first, newest], [duplicate]]):
            results = collector._collect(2)
        self.assertEqual([collector._normalize_link(item.entry_id) for item in results], [
            "https://arxiv.org/abs/2608.00002",
            "https://arxiv.org/abs/2608.00001",
        ])
        self.assertIs(results[1], duplicate)

    def test_collect_prefers_updated_revision_and_canonicalizes_legacy_ids(self):
        collector = self.make_collector()
        new_v1 = paper("2608.00004v1", "Autonomous Driving original", 20, updated_day=20, scheme="http")
        new_v2 = paper("2608.00004v2", "Autonomous Driving revised", 20, updated_day=22)
        legacy_v1 = paper("hep-th/9901001v1", "Autonomous Driving legacy original", 19, updated_day=19)
        legacy_v2 = paper("hep-th/9901001v2", "Autonomous Driving legacy revised", 19, updated_day=21, scheme="http")
        with patch.object(collector, "_search", side_effect=[[new_v1, legacy_v1], [new_v2, legacy_v2]]):
            results = collector._collect(2)
        self.assertEqual(len(results), 2)
        self.assertIn(new_v2, results)
        self.assertIn(legacy_v2, results)
        self.assertEqual(
            {collector._normalize_link(item.entry_id) for item in results},
            {
                "https://arxiv.org/abs/2608.00004",
                "https://arxiv.org/abs/hep-th/9901001",
            },
        )

    def test_collect_orders_tied_publication_dates_by_canonical_id(self):
        collector = self.make_collector()
        higher_id = paper("2608.00011v1", "Autonomous Driving higher", 20)
        lower_id = paper("2608.00010v1", "Autonomous Driving lower", 20)
        with patch.object(collector, "_search", side_effect=[[higher_id], [lower_id]]):
            results = collector._collect(1)
        self.assertEqual(
            [collector._normalize_link(item.entry_id) for item in results],
            ["https://arxiv.org/abs/2608.00010", "https://arxiv.org/abs/2608.00011"],
        )

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

    def test_initialize_caps_total_new_papers_after_existing_filtering(self):
        collector = ArxivCollector(
            self.papers_path,
            init_results=10,
            daily_results=10,
            daily_total_limit=5,
            topic_queries={"one": "q1"},
        )
        existing = paper("2608.00020v1", "Existing Autonomous Driving", 20)
        Path(self.papers_path).write_text(
            "| 日期 | 标题 | 链接 | 简要总结 |\n"
            "| --- | --- | --- | --- |\n"
            + collector._format_row(existing),
            encoding="utf-8",
        )
        candidates = [
            paper("2608.00020v2", "Existing Autonomous Driving revised", 21),
            *[
                paper(
                    f"2608.{identifier:05d}v1",
                    f"Autonomous Driving {identifier}",
                    identifier,
                )
                for identifier in range(7, 0, -1)
            ],
        ]

        with patch.object(collector, "_collect", return_value=candidates) as collect:
            count = collector.initialize()

        collect.assert_called_once_with(10)
        content = Path(self.papers_path).read_text(encoding="utf-8")
        self.assertEqual(count, 5)
        for identifier in range(7, 2, -1):
            self.assertIn(f"https://arxiv.org/abs/2608.{identifier:05d}", content)
        self.assertNotIn("https://arxiv.org/abs/2608.00002", content)
        self.assertNotIn("https://arxiv.org/abs/2608.00001", content)
        self.assertEqual(content.count("https://arxiv.org/abs/2608.00020"), 1)

    def test_run_daily_respects_configured_total_limit(self):
        collector = ArxivCollector(
            self.papers_path,
            init_results=10,
            daily_results=10,
            daily_total_limit=2,
            topic_queries={"one": "q1"},
        )
        candidates = [
            paper(
                f"2608.{identifier:05d}v1",
                f"Autonomous Driving {identifier}",
                identifier,
            )
            for identifier in range(4, 0, -1)
        ]

        with patch.object(collector, "_collect", return_value=candidates) as collect:
            count = collector.run_daily()

        collect.assert_called_once_with(10)
        content = Path(self.papers_path).read_text(encoding="utf-8")
        self.assertEqual(count, 2)
        self.assertIn("https://arxiv.org/abs/2608.00004", content)
        self.assertIn("https://arxiv.org/abs/2608.00003", content)
        self.assertNotIn("https://arxiv.org/abs/2608.00002", content)
        self.assertNotIn("https://arxiv.org/abs/2608.00001", content)

    def test_daily_total_limit_defaults_to_five_and_reads_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            collector = ArxivCollector(self.papers_path, topic_queries={"one": "q1"})
        self.assertEqual(collector.daily_total_limit, 5)

        with patch.dict(os.environ, {"ARXIV_DAILY_TOTAL_LIMIT": "3"}, clear=True):
            collector = ArxivCollector(self.papers_path, topic_queries={"one": "q1"})
        self.assertEqual(collector.daily_total_limit, 3)

    def test_existing_and_rendered_links_are_canonical(self):
        collector = self.make_collector()
        Path(self.papers_path).write_text(
            "| 日期 | 标题 | 链接 | 简要总结 |\n"
            "| --- | --- | --- | --- |\n"
            "| 1999-01-01 | Legacy | http://arxiv.org/abs/hep-th/9901001v3 | pending |\n",
            encoding="utf-8",
        )
        self.assertEqual(
            collector._load_existing_links(),
            {"hep-th/9901001"},
        )
        rendered = collector._format_row(
            paper("2608.00012v4", "Autonomous Driving render", 20, scheme="http")
        )
        self.assertIn("https://arxiv.org/abs/2608.00012", rendered)
        self.assertNotIn("v4", rendered)

    def test_normalize_link_accepts_urls_and_bare_ids(self):
        collector = self.make_collector()
        cases = {
            "http://arxiv.org/abs/2608.00013v2": "https://arxiv.org/abs/2608.00013",
            "https://arxiv.org/abs/2608.00013v3": "https://arxiv.org/abs/2608.00013",
            "2608.00013v4": "https://arxiv.org/abs/2608.00013",
            "http://arxiv.org/abs/hep-th/9901001v2": "https://arxiv.org/abs/hep-th/9901001",
            "hep-th/9901001v3": "https://arxiv.org/abs/hep-th/9901001",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(collector._normalize_link(value), expected)

    def test_constructor_rejects_invalid_collection_limits(self):
        for keyword, value in (
            ("init_results", 0),
            ("init_results", -1),
            ("daily_results", 0),
            ("daily_results", -1),
            ("daily_total_limit", 0),
            ("daily_total_limit", -1),
            ("daily_total_limit", 6),
        ):
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises(ValueError):
                    ArxivCollector(self.papers_path, topic_queries={"one": "q1"}, **{keyword: value})

        for environment in (
            {"ARXIV_INIT_RESULTS": "0"},
            {"ARXIV_DAILY_RESULTS": "0"},
            {"ARXIV_DAILY_TOTAL_LIMIT": "0"},
            {"ARXIV_DAILY_TOTAL_LIMIT": "6"},
            {"ARXIV_PAGE_SIZE": "0"},
            {"ARXIV_PAGE_SIZE": "2001"},
            {"ARXIV_MAX_RETRIES": "0"},
        ):
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValueError):
                        ArxivCollector(self.papers_path, topic_queries={"one": "q1"})

    def test_search_and_collect_reject_zero_max_results(self):
        collector = self.make_collector()
        with self.assertRaises(ValueError):
            collector._search("q1", 0)
        with self.assertRaises(ValueError):
            collector._collect(0)

    def test_collect_reuses_one_arxiv_client_for_all_topics(self):
        with patch("arxiv_crawler.arxiv.Client") as client_class:
            client_class.return_value.results.side_effect = [[], []]
            collector = self.make_collector()
            self.assertEqual(collector._collect(1), [])
        client_class.assert_called_once()
        self.assertEqual(client_class.return_value.results.call_count, 2)


if __name__ == "__main__":
    unittest.main()
