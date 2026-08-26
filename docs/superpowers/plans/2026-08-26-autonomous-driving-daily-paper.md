# Autonomous Driving Daily Paper Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 VLA/WAM 每日论文站改造成覆盖自动驾驶全栈方向的每日 arXiv 论文站，并推送到用户的 GitHub fork 部署 Pages。

**Architecture:** 用独立主题配置维护七组 arXiv 查询，爬虫逐组执行、容错、相关性过滤并按规范化 arXiv ID 合并。保持 `papers.md` 四列协议和现有静态站生成链路不变，只调整摘要提示词、自动驾驶品牌文案、自动化配置和初始数据。

**Tech Stack:** Python 3.9、`arxiv`、DeepSeek 官方 OpenAI 兼容 API、原生 HTML/CSS/JavaScript、Playwright、GitHub Actions、GitHub Pages。

## Global Constraints

- 首页布局、配色、日期分组、论文卡片、搜索、详情页、首图和阅读进度保持现有站点行为。
- 前台继续展示统一论文流，不增加分类标签或分类页面。
- 检索必须覆盖综合、感知、定位与建图、预测、规划与决策、控制、端到端/基础模型七组方向。
- 清除全部旧 VLA/WAM 论文表格、详情页、封面和图片缓存，然后用自动驾驶论文重建。
- DeepSeek 令牌只能来自本地环境变量或 GitHub Secret `DEEPSEEK_API_KEY`，不得写入仓库或日志。
- 摘要请求必须直连 `https://api.deepseek.com` 并默认使用 `deepseek-v4-flash`；不得保留 ModelScope 或多供应商回退。
- 单组检索失败不能中止其余主题；全部主题失败必须返回失败。
- 默认首次每组 80 条、每日每组 10 条，允许环境变量覆盖。
- Python 代码兼容工作流指定的 Python 3.9，不使用 `X | None` 等 3.10 才支持的注解。

---

## File Structure

- Create `scripts/autonomous_driving_topics.py`: 七组查询、允许类别和相关性判定的唯一来源。
- Modify `scripts/arxiv_crawler.py`: 多主题执行、跨主题去重、排序、部分失败容错和空库初始化判断。
- Create `tests/test_arxiv_crawler.py`: 查询配置、相关性、去重、排序和失败语义测试。
- Modify `scripts/generate_summaries.py`: 自动驾驶专用结构化摘要提示词和 DeepSeek 官方 API 客户端。
- Modify `test_api.py`: DeepSeek Flash 手工连通性测试。
- Modify `scripts/build_site.py`: 自动驾驶品牌、SEO、英雄区和搜索示例。
- Modify `scripts/modern_ui.css`: 英雄区背景水印从 `VLA` 改为 `AD`。
- Modify `scripts/fetch_paper_images.py`: 抓图 User-Agent 更名。
- Modify `package.json`, `package-lock.json`: 包名改为 `daily-arxiv-autodrive`。
- Create `tests/test_branding.py`: 摘要提示词和生成 HTML 的品牌回归测试。
- Create `tests/test_deepseek_config.py`: DeepSeek 地址、密钥、Flash 模型和无回退行为测试。
- Modify `.env.example`, `.github/workflows/deploy.yml`, `README.md`: DeepSeek 配置、新抓取默认参数、无令牌可部署行为和使用说明。
- Modify `papers.md`, generated `site/**`: 清空旧数据，实时回填自动驾驶论文并重建站点。

---

### Task 1: Add topic registry and multi-query collector

**Files:**
- Create: `scripts/autonomous_driving_topics.py`
- Modify: `scripts/arxiv_crawler.py:1-253`
- Create: `tests/__init__.py`
- Create: `tests/test_arxiv_crawler.py`

**Interfaces:**
- Produces: `get_topic_queries(query_override: Optional[str]) -> Dict[str, str]`
- Produces: `is_relevant_paper(title: str, summary: str) -> bool`
- Produces: `ArxivCollector._collect(max_results: int) -> List[arxiv.Result]`
- Produces: `ArxivCollector.has_existing_papers() -> bool`

- [ ] **Step 1: Write failing collector tests**

Create `tests/__init__.py` as an empty package marker and create `tests/test_arxiv_crawler.py` with:

```python
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
```

- [ ] **Step 2: Run tests and confirm the missing module/interface failures**

Run:

```powershell
python -m unittest tests.test_arxiv_crawler -v
```

Expected: FAIL because `autonomous_driving_topics` and the multi-query interfaces do not exist.

- [ ] **Step 3: Add the complete topic registry**

Create `scripts/autonomous_driving_topics.py` with:

```python
from typing import Dict, Optional

DRIVING_CONTEXT = (
    '(all:"autonomous driving" OR all:"self-driving" OR all:"automated driving" OR '
    '(all:"autonomous vehicle" AND (all:"road" OR all:"traffic" OR all:"driving")) OR '
    'all:"driving scene" OR all:"road vehicle")'
)

TOPIC_QUERIES: Dict[str, str] = {
    "overview": (
        'all:"autonomous driving" OR all:"self-driving" OR all:"automated driving" OR '
        '(all:"autonomous vehicle" AND (all:"road" OR all:"traffic" OR all:"driving"))'
    ),
    "perception": DRIVING_CONTEXT + " AND " + (
        '(all:"perception" OR all:"BEV" OR all:"3D object detection" OR all:"object tracking" OR '
        'all:"lane detection" OR all:"occupancy prediction" OR all:"sensor fusion")'
    ),
    "localization_mapping": DRIVING_CONTEXT + " AND " + (
        '(all:"localization" OR all:"mapping" OR all:"SLAM" OR all:"visual odometry" OR all:"HD map")'
    ),
    "prediction": DRIVING_CONTEXT + " AND " + (
        '(all:"motion forecasting" OR all:"trajectory prediction" OR all:"behavior prediction")'
    ),
    "planning_decision": DRIVING_CONTEXT + " AND " + (
        '(all:"motion planning" OR all:"trajectory planning" OR all:"decision making" OR all:"driving policy")'
    ),
    "control": DRIVING_CONTEXT + " AND " + (
        '(all:"vehicle control" OR all:"model predictive control" OR all:"path tracking")'
    ),
    "end_to_end_foundation": (
        'all:"end-to-end autonomous driving" OR all:"end-to-end driving" OR '
        'all:"foundation model" AND ' + DRIVING_CONTEXT + ' OR '
        'all:"vision language model" AND ' + DRIVING_CONTEXT + ' OR '
        'all:"world model" AND ' + DRIVING_CONTEXT
    ),
}

ALLOWED_PRIMARY_CATEGORIES = {
    "cs.CV", "cs.RO", "cs.AI", "cs.LG", "cs.MM", "eess.SY", "eess.IV"
}

STRONG_DRIVING_TERMS = (
    "autonomous driving", "self-driving", "self driving", "automated driving",
    "end-to-end driving", "end to end driving", "road vehicle",
)
DRIVING_CONTEXT_TERMS = ("driving", "road", "traffic", "automotive")
TASK_TERMS = (
    "perception", "bev", "object detection", "tracking", "lane", "occupancy",
    "sensor fusion", "localization", "mapping", "slam", "odometry", "hd map",
    "motion forecasting", "trajectory prediction", "behavior prediction", "planning",
    "decision making", "driving policy", "vehicle control", "path tracking",
    "foundation model", "vision language model", "world model",
)


def get_topic_queries(query_override: Optional[str] = None) -> Dict[str, str]:
    if query_override and query_override.strip():
        return {"custom": query_override.strip()}
    return dict(TOPIC_QUERIES)


def is_relevant_paper(title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()
    if any(term in text for term in STRONG_DRIVING_TERMS):
        return True
    has_context = any(term in text for term in DRIVING_CONTEXT_TERMS)
    has_task = any(term in text for term in TASK_TERMS)
    return has_context and has_task
```

- [ ] **Step 4: Refactor the collector around the topic registry**

In `scripts/arxiv_crawler.py`, import `Dict` and `Optional`, import the registry interfaces, set defaults to 80/10 per topic, and replace the single-query search path with these complete methods:

```python
from autonomous_driving_topics import (
    ALLOWED_PRIMARY_CATEGORIES,
    get_topic_queries,
    is_relevant_paper,
)


def __init__(self, papers_path: str, init_results: int = None, daily_results: int = None,
             query_keyword: str = None, topic_queries: Optional[Dict[str, str]] = None):
    self.papers_path = papers_path
    self.init_results = init_results or int(os.getenv("ARXIV_INIT_RESULTS", "80"))
    self.daily_results = daily_results or int(os.getenv("ARXIV_DAILY_RESULTS", "10"))
    override = query_keyword or os.getenv("ARXIV_QUERY_KEYWORD")
    self.topic_queries = topic_queries or get_topic_queries(override)
    self.arxiv_page_size = int(os.getenv("ARXIV_PAGE_SIZE", "20"))
    self.arxiv_delay_seconds = float(os.getenv("ARXIV_DELAY_SECONDS", "10"))


def _search(self, query: str, max_results: int) -> List[arxiv.Result]:
    max_retries = int(os.getenv("ARXIV_MAX_RETRIES", "3"))
    retry_base_seconds = float(os.getenv("ARXIV_RETRY_BASE_SECONDS", "30"))
    page_size = max(1, min(max_results, self.arxiv_page_size))
    client = arxiv.Client(page_size=page_size, delay_seconds=self.arxiv_delay_seconds, num_retries=0)
    last_error = None
    for attempt in range(max_retries):
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            return list(client.results(search))
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait_seconds = retry_base_seconds * (2 ** attempt)
                print(f"arXiv 搜索失败，{wait_seconds:.0f} 秒后重试 {attempt + 1}/{max_retries}: {exc!r}")
                time.sleep(wait_seconds)
    raise RuntimeError(f"arXiv 搜索达到最大重试次数: {last_error!r}")


def _collect(self, max_results: int) -> List[arxiv.Result]:
    merged: Dict[str, arxiv.Result] = {}
    failed_topics = []
    for topic_name, query in self.topic_queries.items():
        print(f"检索主题: {topic_name}")
        try:
            results = self._search(query, max_results)
        except Exception as exc:
            failed_topics.append(topic_name)
            print(f"主题检索失败，继续其他主题: {topic_name}: {exc!r}")
            continue
        for result in results:
            if result.primary_category not in self._ALLOWED_PRIMARY_CATEGORIES:
                continue
            if not is_relevant_paper(result.title or "", result.summary or ""):
                continue
            normalized_link = self._normalize_link(result.entry_id)
            current = merged.get(normalized_link)
            if current is None or result.published > current.published:
                merged[normalized_link] = result
    if len(failed_topics) == len(self.topic_queries):
        raise RuntimeError(f"全部 {len(self.topic_queries)} 个自动驾驶主题检索失败")
    return sorted(merged.values(), key=lambda result: result.published, reverse=True)


def has_existing_papers(self) -> bool:
    return bool(self._load_existing_links())
```

Set `_ALLOWED_PRIMARY_CATEGORIES = ALLOWED_PRIMARY_CATEGORIES`. Update `initialize()` and `run_daily()` to use `self._collect(self.init_results)` and `self._collect(self.daily_results)` respectively. In `__main__`, choose daily mode with `collector.has_existing_papers()` instead of checking whether the header-only file has nonzero bytes.

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
python -m unittest tests.test_arxiv_crawler -v
python -m unittest discover -s tests -v
```

Expected: all collector tests PASS.

- [ ] **Step 6: Commit the collector slice**

```powershell
git add scripts/autonomous_driving_topics.py scripts/arxiv_crawler.py tests/__init__.py tests/test_arxiv_crawler.py
git commit -m "feat: collect autonomous driving papers by topic"
```

---

### Task 2: Retheme summaries and generated pages

**Files:**
- Modify: `scripts/generate_summaries.py:14-260`
- Modify: `scripts/build_site.py:39-145,790-850`
- Modify: `scripts/modern_ui.css:150-165`
- Modify: `scripts/fetch_paper_images.py:164`
- Modify: `package.json:2`
- Modify: `package-lock.json:2,7`
- Create: `tests/test_branding.py`

**Interfaces:**
- Produces: `AUTONOMOUS_DRIVING_SUMMARY_PROMPT: str`
- Produces: `get_arxiv_keyword_label() -> str`, defaulting to `自动驾驶` and reading optional `ARXIV_KEYWORD_LABEL`.

- [ ] **Step 1: Add failing branding tests**

Create `tests/test_branding.py`:

```python
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
        self.assertIn("自动驾驶每日论文卡", html)
        self.assertIn("Autonomous Driving Research Feed", html)
        self.assertIn("UniAD、BEVFormer、nuScenes", html)
        self.assertNotIn("VLA/WAM", html)
        self.assertNotIn("World Action Model Feed", html)

    def test_keyword_label_has_safe_display_override(self):
        with patch.dict(os.environ, {"ARXIV_KEYWORD_LABEL": "智能驾驶"}, clear=True):
            self.assertEqual(build_site.get_arxiv_keyword_label(), "智能驾驶")

    def test_summary_prompt_requests_driving_evidence(self):
        prompt = generate_summaries.AUTONOMOUS_DRIVING_SUMMARY_PROMPT
        for phrase in ("自动驾驶", "传感器", "数据集", "评估指标", "闭环"):
            self.assertIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm branding failures**

Run `python -m unittest tests.test_branding -v`.

Expected: FAIL because the old VLA/WAM strings remain and the prompt constant is missing.

- [ ] **Step 3: Extract an autonomous-driving summary prompt**

Add this constant near the top of `scripts/generate_summaries.py` and pass it as the system message content:

```python
AUTONOMOUS_DRIVING_SUMMARY_PROMPT = """你是一名自动驾驶论文阅读专家。只能根据提供的 arXiv 论文 HTML 原文生成中文结构化总结，不得补造论文没有报告的实验信息。

严格使用 Markdown 二级标题和无序列表，输出以下六个部分：

## 研究单位
- 列出论文作者所属机构。

## 论文概述
- 说明论文解决的自动驾驶任务、应用场景和研究问题。

## 核心贡献
- 用 3-5 个要点概括可由原文支持的贡献。

## 方法描述
- 说明所属模块、输入传感器、场景表示、模型结构和关键技术。
- 若原文未报告某项信息，明确写“原文未报告”。

## 数据集与资源
- 列出数据集、仿真器、模型规模、训练资源和实车平台；未报告的信息明确标注。

## 评估与结果
- 列出评估基准、评估指标、关键数值、对比方法，以及开环、闭环、仿真或实车条件。

每个标题后换行并使用“- ”列表项，不要输出代码块或思考过程。"""
```

- [ ] **Step 4: Replace generated-site branding without changing layout**

In `scripts/build_site.py`, replace query-derived display labels with:

```python
DEFAULT_ARXIV_KEYWORD_LABEL = "自动驾驶"


def get_arxiv_keyword_label() -> str:
    return (os.getenv("ARXIV_KEYWORD_LABEL") or DEFAULT_ARXIV_KEYWORD_LABEL).strip()
```

In `generate_index_html()`, keep the existing HTML hierarchy and classes while changing exact copy to:

```html
<span class="site-brand-mark">AD</span>
<span>Research Brief</span>
<p class="eyebrow">Autonomous Driving Research Feed</p>
<p class="site-subtitle">聚合感知、定位、预测、规划、控制与端到端驾驶最新论文，提炼核心贡献、方法与实验结果。</p>
<input id="search" type="search" placeholder="比如：UniAD、BEVFormer、nuScenes、motion planning..." aria-label="搜索论文卡片" />
```

Change `scripts/modern_ui.css` watermark to `content: "AD";`, change the image-fetcher User-Agent to `daily-arxiv-autodrive/paper-image-fetcher (+https://arxiv.org)`, and rename the package to `daily-arxiv-autodrive` in both package files.

- [ ] **Step 5: Run branding and collector tests**

```powershell
python -m unittest tests.test_branding -v
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the presentation slice**

```powershell
git add scripts/generate_summaries.py scripts/build_site.py scripts/modern_ui.css scripts/fetch_paper_images.py package.json package-lock.json tests/test_branding.py
git commit -m "feat: retheme paper site for autonomous driving"
```

---

### Task 3: Migrate summaries to DeepSeek Flash and update automation/docs

**Files:**
- Modify: `scripts/generate_summaries.py`
- Modify: `test_api.py`
- Create: `tests/test_deepseek_config.py`
- Modify: `.env.example`
- Modify: `.github/workflows/deploy.yml`
- Modify: `README.md`

**Interfaces:**
- Produces: `DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"`
- Produces: `DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"`
- Produces: `get_client() -> OpenAI` using only `DEEPSEEK_API_KEY` and the DeepSeek official endpoint.
- Produces: `get_model() -> str` with the Flash default and optional `DEEPSEEK_MODEL` override.
- Consumes: multi-query defaults from `scripts/autonomous_driving_topics.py`.
- Produces: a workflow that safely deploys pending summaries when `DEEPSEEK_API_KEY` is absent.

- [ ] **Step 1: Add failing DeepSeek-only configuration tests**

Create `tests/test_deepseek_config.py` with mocked environment/client tests that assert:

```python
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_summaries


class DeepSeekConfigTests(unittest.TestCase):
    def test_defaults_are_official_flash(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            self.assertEqual(generate_summaries.get_model(), "deepseek-v4-flash")
            with patch.object(generate_summaries, "OpenAI") as openai_class:
                generate_summaries.get_client()
        openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )

    def test_missing_deepseek_key_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                generate_summaries.get_client()

    def test_model_override_stays_with_deepseek_configuration(self):
        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "deepseek-v4-flash"}, clear=True):
            self.assertEqual(generate_summaries.get_model(), "deepseek-v4-flash")

    def test_no_modelscope_configuration_remains(self):
        source = (ROOT / "scripts" / "generate_summaries.py").read_text(encoding="utf-8")
        self.assertNotIn("MODELSCOPE", source)
        self.assertNotIn("api-inference.modelscope.cn", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Replace ModelScope and model fallback with one DeepSeek model**

In `scripts/generate_summaries.py`, define:

```python
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL).strip()
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model() -> str:
    return (os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL).strip()
```

Remove `MODELSCOPE_*`, `MODELSCOPE_MODELS`, `RATE_LIMITED_MODELS`, model-list iteration and cross-model fallback. `generate_summary_for_link()` must make retry attempts against exactly `model or get_model()` and return an empty string after that one model exhausts its retries. Preserve the automatic-driving prompt, HTML retry logic, output cleanup and batch writes.

Update `test_api.py` to read `DEEPSEEK_API_KEY`, use `https://api.deepseek.com`, and call `deepseek-v4-flash`; it must fail clearly before network access when the key is missing.

- [ ] **Step 3: Update example configuration**

Use these provider and crawler blocks in `.env.example`:

```dotenv
# DeepSeek 官方 API 配置
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

# arXiv 爬虫配置：不设置 ARXIV_QUERY_KEYWORD 时启用七组自动驾驶检索
ARXIV_KEYWORD_LABEL=自动驾驶
ARXIV_INIT_RESULTS=80
ARXIV_DAILY_RESULTS=10
ARXIV_PAGE_SIZE=20
ARXIV_DELAY_SECONDS=10
ARXIV_RETRY_BASE_SECONDS=30
ARXIV_MAX_RETRIES=3
# ARXIV_QUERY_KEYWORD=all:"your custom query"
```

- [ ] **Step 4: Make GitHub Actions use DeepSeek Flash and topic defaults**

The test, crawler and summary steps must be:

```yaml
      - name: 运行单元测试
        run: python -m unittest discover -s tests -v

      - name: 运行 arXiv 爬虫
        run: python scripts/arxiv_crawler.py
        env:
          ARXIV_INIT_RESULTS: 80
          ARXIV_DAILY_RESULTS: 10
          ARXIV_PAGE_SIZE: 10
          ARXIV_DELAY_SECONDS: 15
          ARXIV_RETRY_BASE_SECONDS: 60
          ARXIV_MAX_RETRIES: 4

      - name: 使用 DeepSeek Flash 生成论文摘要
        run: |
          if [ -z "$DEEPSEEK_API_KEY" ]; then
            echo "未配置 DEEPSEEK_API_KEY，本次保留待生成摘要并继续部署"
            exit 0
          fi
          python scripts/generate_summaries.py
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
          DEEPSEEK_BASE_URL: https://api.deepseek.com
          DEEPSEEK_MODEL: deepseek-v4-flash
```

Remove `ARXIV_QUERY_KEYWORD` from crawler/build and remove every `MODELSCOPE_*` workflow variable.

- [ ] **Step 5: Rewrite README around autonomous driving and DeepSeek**

Document the seven automatic-driving topics, per-topic merging/deduplication, direct DeepSeek traffic path, `DEEPSEEK_API_KEY`, official base URL, `deepseek-v4-flash`, custom arXiv override, unchanged image/build/GA/Pages commands, and the behavior when the key is absent. State explicitly that the OpenAI Python package is only an API-compatible client and requests go directly to DeepSeek.

- [ ] **Step 6: Validate provider isolation, automation and tests**

Run:

```powershell
python -m unittest tests.test_deepseek_config -v
python -m unittest discover -s tests -v
rg -n -i 'MODELSCOPE|api-inference\.modelscope\.cn|ARXIV_QUERY_KEYWORD.*VLA|VLA/WAM|World Action Model Feed|OpenVLA' .env.example README.md .github scripts test_api.py package.json package-lock.json
rg -n 'DEEPSEEK_API_KEY|https://api\.deepseek\.com|deepseek-v4-flash' .env.example README.md .github scripts test_api.py
```

Expected: all tests PASS; the first `rg` returns no production matches; the second confirms the exact DeepSeek key, endpoint and Flash model. Do not make a live API request unless `DEEPSEEK_API_KEY` is already configured, and never print its value.

- [ ] **Step 7: Commit provider, automation and documentation together**

```powershell
git add scripts/generate_summaries.py test_api.py tests/test_deepseek_config.py .env.example .github/workflows/deploy.yml README.md
git commit -m "feat: use DeepSeek Flash for paper summaries"
```

---

### Task 4: Remove old content and build a real autonomous-driving dataset

**Files:**
- Modify: `papers.md`
- Remove and regenerate: `site/papers/**`, `site/covers/**`
- Remove: `site/assets/paper-images/**`
- Modify: `site/assets/paper-images.json`, `site/assets/data.json`, `site/index.html`, generated site assets

**Interfaces:**
- Consumes: `ArxivCollector.initialize()` with seven topic queries.
- Produces: a nonempty `papers.md` containing only the new automatic-driving collection and a matching static site.

- [ ] **Step 1: Verify the exact generated-data deletion targets**

Run this read-only validation before removal:

```powershell
$workspaceRoot = (Resolve-Path -LiteralPath '.').Path
$targets = @('site\papers', 'site\covers', 'site\assets\paper-images')
foreach ($target in $targets) {
    $resolved = (Resolve-Path -LiteralPath $target).Path
    if (-not $resolved.StartsWith((Join-Path $workspaceRoot 'site') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing target outside site: $resolved"
    }
    Write-Output $resolved
}
```

Expected: exactly the three directories under the current workspace's `site` directory.

- [ ] **Step 2: Remove only the approved VLA/WAM generated content**

After Step 1 succeeds, use native PowerShell removal on the same literal paths:

```powershell
Remove-Item -LiteralPath 'site\papers' -Recurse -Force
Remove-Item -LiteralPath 'site\covers' -Recurse -Force
Remove-Item -LiteralPath 'site\assets\paper-images' -Recurse -Force
```

Reset `papers.md` to exactly:

```markdown
| 日期 | 标题 | 链接 | 简要总结 |
| --- | --- | --- | --- |
```

Reset `site/assets/paper-images.json` to `{}`.

- [ ] **Step 3: Run the full initial arXiv backfill**

Run:

```powershell
Remove-Item Env:ARXIV_QUERY_KEYWORD -ErrorAction SilentlyContinue
$env:ARXIV_INIT_RESULTS = '80'
$env:ARXIV_PAGE_SIZE = '20'
$env:ARXIV_DELAY_SECONDS = '10'
python scripts/arxiv_crawler.py
```

Expected: output lists seven topic names, reports initialization rather than daily mode, and adds at least one paper. If arXiv temporarily rejects one topic, the other topics continue. If all seven fail, do not fabricate data; rerun after the reported retry interval.

- [ ] **Step 4: Generate summaries only when a token is already configured**

```powershell
if (Test-Path Env:DEEPSEEK_API_KEY) {
    python scripts/generate_summaries.py
} else {
    Write-Output 'DEEPSEEK_API_KEY is not configured; summaries remain pending.'
}
```

Expected: the command never prints the token. With no token it only prints the skip message.

- [ ] **Step 5: Fetch initial figures and rebuild the site**

```powershell
python scripts/fetch_paper_images.py --max-items 30
python scripts/build_site.py
```

Expected: `site/index.html` and `site/assets/data.json` exist; `site/papers/` contains one directory per collected paper; at most 30 initial figure fetches are attempted.

- [ ] **Step 6: Validate record count, IDs and generated-page consistency**

Run:

```powershell
python -c "import json,re,pathlib; p=pathlib.Path('papers.md').read_text(encoding='utf-8'); ids=re.findall(r'arxiv.org/abs/([0-9.]+)',p); data=json.loads(pathlib.Path('site/assets/data.json').read_text(encoding='utf-8')); assert ids; assert len(ids)==len(set(ids)); assert len(data)==len(ids); print({'papers':len(ids),'site_records':len(data)})"
python -m unittest discover -s tests -v
git diff --check
```

Expected: nonzero equal counts, unique normalized IDs, all tests PASS, and no whitespace errors.

- [ ] **Step 7: Commit the clean autonomous-driving dataset**

```powershell
git add papers.md site
git commit -m "data: initialize autonomous driving paper library"
```

---

### Task 5: Browser and content QA

**Files:**
- Inspect: `site/index.html`, `site/papers/**/index.html`
- Generate locally without committing: `artifacts/qa-home-desktop.png`, `artifacts/qa-home-mobile.png`

**Interfaces:**
- Consumes: the fully generated `site/` directory.
- Produces: visual evidence that layout remains equivalent to the supplied VLA/WAM screenshot with automatic-driving copy.

- [ ] **Step 1: Install the locked browser dependency**

Run `npm ci`.

Expected: Playwright 1.58.2 installs from `package-lock.json` without changing the lockfile.

- [ ] **Step 2: Start a hidden local server and capture desktop/mobile pages**

```powershell
New-Item -ItemType Directory -Path 'artifacts' -Force | Out-Null
$server = Start-Process python -ArgumentList '-m','http.server','8000','--directory','site' -WindowStyle Hidden -PassThru
try {
    npx playwright screenshot --device="Desktop Chrome HiDPI" --full-page http://127.0.0.1:8000 artifacts/qa-home-desktop.png
    npx playwright screenshot --device="iPhone 13" --full-page http://127.0.0.1:8000 artifacts/qa-home-mobile.png
} finally {
    Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
}
```

Expected: both PNG files exist and the server process is stopped.

- [ ] **Step 3: Inspect the screenshots and generated HTML**

Confirm visually that the hero/search two-column layout, date groups, cards, figures and responsive mobile stacking match the original design. Confirm the visible title is “自动驾驶每日论文卡”, the brand mark is `AD`, and no VLA/WAM hero copy remains.

- [ ] **Step 4: Run final source, secret and Git checks**

```powershell
rg -n -i 'VLA/WAM|World Action Model Feed|OpenVLA|all:"VLA"' README.md .env.example .github scripts package.json package-lock.json site/index.html
rg --pcre2 -n 'DEEPSEEK_API_KEY\s*=\s*(?!your_api_key_here)' -g '*.env' -g '*.example' .
python -m unittest discover -s tests -v
git diff --check
git status --short
```

Expected: no old-brand matches, no real-token match, all tests PASS, no whitespace errors, and only intentionally untracked `artifacts/` screenshots remain. Do not add `artifacts/` to the commit.

- [ ] **Step 5: Commit only if visual QA required source fixes**

If a source fix was necessary, stage only the named source files, rerun Step 4, and commit with:

```powershell
git commit -m "fix: polish autonomous driving paper site"
```

---

### Task 6: Push the approved result and verify Pages

**Files:**
- No source edits expected.
- Mutates: Git remotes, the user's GitHub fork, GitHub Actions, and GitHub Pages.

**Interfaces:**
- Consumes: a clean local `master` branch with all prior tests passing.
- Produces: the fork commit URL, Actions run URL, and Pages URL.

- [ ] **Step 1: Verify local handoff state**

```powershell
git status --short --branch
git log -6 --oneline --decorate
git remote -v
```

Expected: no tracked changes, local commits are ahead of `Infinity4B/daily-arxiv-vla`, and the current source remote still points to the upstream repository.

- [ ] **Step 2: Create or open the user's fork and validate its HTTPS URL**

Open `https://github.com/Infinity4B/daily-arxiv-vla/fork`, create the fork under the user's authenticated account if it does not already exist, and copy the fork's HTTPS clone URL. Validate the copied URL before changing remotes:

```powershell
$forkUrl = (Get-Clipboard).Trim()
if ($forkUrl -notmatch '^https://github\.com/[^/]+/daily-arxiv-vla(?:\.git)?$') {
    throw "Clipboard does not contain a daily-arxiv-vla fork HTTPS URL"
}
$forkUrl
```

Expected: a URL under the user's account, not `Infinity4B/daily-arxiv-vla` unless that is the authenticated account the user controls.

- [ ] **Step 3: Preserve upstream, set the fork as origin and push**

```powershell
git remote rename origin upstream
git remote add origin $forkUrl
git ls-remote origin HEAD
git push -u origin master
```

Expected: `origin/master` points to the final local commit. If Git Credential Manager opens a browser, complete that one-time authorization; never paste or print a personal access token into the terminal.

- [ ] **Step 4: Enable and verify GitHub Pages**

In the fork's `Settings -> Pages`, select `GitHub Actions` as the source. In `Actions`, open the workflow run created by the push and verify test, crawl, build, upload and deploy steps. If `DEEPSEEK_API_KEY` is absent, verify that the summary step reports the documented skip and that deployment still succeeds.

- [ ] **Step 5: Verify the public artifact and report exact evidence**

Open the deployed Pages URL shown by the workflow and check that the title, automatic-driving papers, search and one paper detail page load. Report:

- fork repository URL
- pushed commit SHA
- GitHub Actions run URL and conclusion
- GitHub Pages URL
- collected paper count
- summary count versus pending count
- local desktop/mobile screenshot paths

Do not claim deployment complete until the public Pages URL returns the generated automatic-driving site.
