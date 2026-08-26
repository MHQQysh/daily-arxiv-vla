# 100-Paper Backfill and Daily-20 Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill the existing five-paper autonomous-driving library to exactly 100 papers with restart-safe DeepSeek summaries and paper images, then add at most 20 papers per scheduled day.

**Architecture:** Split initialization and daily write limits in the crawler, and add an explicit total-library backfill command that computes only the missing count. Keep summaries and images incremental: successful entries are saved immediately and subsequent runs select only incomplete entries. Preserve five-way DeepSeek concurrency while allowing up to 100 pending summaries in the one-time backfill.

**Tech Stack:** Python 3.9+, `arxiv`, `requests`, OpenAI-compatible DeepSeek client, Pillow, Node.js 20, Playwright, GitHub Actions, static GitHub Pages.

## Global Constraints

- Initial target is exactly 100 total papers; the existing 5 count toward that target.
- Scheduled updates append at most 20 papers per day and never delete historical papers.
- DeepSeek model remains `deepseek-v4-flash` at `https://api.deepseek.com`.
- DeepSeek concurrency never exceeds 5 workers.
- arXiv requests retain a 10-second HTTP timeout, two attempts, five-second retry base, and live unbuffered logging.
- Push and default manual workflow runs remain non-writing crawler previews.
- Never print, commit, or upload the local DeepSeek API key.

---

### Task 1: Split crawler limits and add exact-total backfill

**Files:**
- Modify: `scripts/arxiv_crawler.py`
- Test: `tests/test_arxiv_crawler.py`

**Interfaces:**
- Consumes: existing `ArxivCollector._collect(max_results)`, `_load_existing_links()`, `_build_new_rows(...)`, and `_append_rows(...)`.
- Produces: `ArxivCollector.backfill_to(target_total: int) -> int`, `init_total_limit`, `daily_total_limit`, and CLI option `--backfill-to N`.

- [ ] **Step 1: Write failing tests for separate limits and exact backfill**

```python
def test_backfill_from_five_to_one_hundred_adds_exactly_ninety_five(self):
    collector = ArxivCollector(
        self.papers_path,
        init_results=100,
        daily_results=20,
        init_total_limit=100,
        daily_total_limit=20,
        topic_queries={"one": "q1"},
    )
    existing = [paper(f"2608.{i:05d}v1", f"Existing {i}", 25) for i in range(1, 6)]
    Path(self.papers_path).write_text(
        "| 日期 | 标题 | 链接 | 简要总结 |\n| --- | --- | --- | --- |\n"
        + "".join(collector._format_row(item) for item in existing),
        encoding="utf-8",
    )
    candidates = [paper(f"2607.{i:05d}v1", f"Candidate {i}", 24) for i in range(1, 121)]
    with patch.object(collector, "_collect", return_value=candidates):
        self.assertEqual(collector.backfill_to(100), 95)
    self.assertEqual(len(collector._load_existing_links()), 100)

def test_daily_update_can_write_twenty_but_not_twenty_one(self):
    collector = ArxivCollector(
        self.papers_path,
        init_results=100,
        daily_results=40,
        init_total_limit=100,
        daily_total_limit=20,
        topic_queries={"one": "q1"},
    )
    candidates = [paper(f"2608.{i:05d}v1", f"Daily {i}", 25) for i in range(1, 26)]
    with patch.object(collector, "_collect", return_value=candidates):
        self.assertEqual(collector.run_daily(), 20)
```

- [ ] **Step 2: Run the crawler tests and confirm the new API is missing**

Run: `py -3.13 -m unittest tests.test_arxiv_crawler -v`

Expected: failure because `init_total_limit` and `backfill_to` do not yet exist, and the old hard maximum rejects 20.

- [ ] **Step 3: Implement separate hard maxima and a limit-aware row builder**

```python
_INIT_TOTAL_LIMIT_HARD_MAX = 100
_DAILY_TOTAL_LIMIT_HARD_MAX = 20

def _build_new_rows(self, results, existing, limit):
    self._require_positive_int("limit", limit, maximum=_INIT_TOTAL_LIMIT_HARD_MAX)
    rows = []
    seen_ids = set(existing)
    for result in results:
        canonical_id = self._canonical_arxiv_id(result.entry_id)
        if canonical_id is None or canonical_id in seen_ids:
            continue
        seen_ids.add(canonical_id)
        rows.append(self._format_row(result))
        if len(rows) >= limit:
            break
    return rows
```

Update the constructor to read `ARXIV_INIT_TOTAL_LIMIT` with maximum 100 and `ARXIV_DAILY_TOTAL_LIMIT` with maximum 20. Make `initialize()` use `init_total_limit`, `run_daily()` and `preview_daily()` use `daily_total_limit`, and pass the limit explicitly to `_build_new_rows`.

- [ ] **Step 4: Implement exact-total backfill and CLI routing**

```python
def backfill_to(self, target_total: int) -> int:
    target = self._require_positive_int(
        "backfill target",
        target_total,
        maximum=_INIT_TOTAL_LIMIT_HARD_MAX,
    )
    self._ensure_md_header()
    existing = self._load_existing_links()
    remaining = target - len(existing)
    if remaining <= 0:
        print(f"论文库已有 {len(existing)} 篇，无需补齐", flush=True)
        return 0
    results = self._collect(self.init_results)
    rows = self._build_new_rows(results, existing, limit=remaining)
    if rows:
        self._append_rows(rows)
    print(f"补齐完成：新增 {len(rows)} 篇，总目标 {target} 篇", flush=True)
    return len(rows)
```

Add `parser.add_argument("--backfill-to", type=int)` and route it before `--dry-run` and normal initialize/daily behavior.

- [ ] **Step 5: Run tests and commit the crawler change**

Run: `py -3.13 -m unittest tests.test_arxiv_crawler -v`

Expected: all crawler tests pass, including exact 5-to-100 backfill and daily-20 enforcement.

```powershell
git add scripts/arxiv_crawler.py tests/test_arxiv_crawler.py
git commit -m "feat: support 100-paper backfill and daily twenty"
```

---

### Task 2: Allow 100 restart-safe summaries with five workers

**Files:**
- Modify: `scripts/generate_summaries.py`
- Test: `tests/test_deepseek_config.py`

**Interfaces:**
- Consumes: existing pending-entry detection and `update_entry_with_summary(...)` checkpoint writes.
- Produces: `get_summary_item_limit() -> int`, hard item maximum 100, and hard worker maximum 5.

- [ ] **Step 1: Write failing tests for 100 items and five-way concurrency**

```python
def test_summary_item_limit_allows_one_hundred(self):
    with patch.dict(os.environ, {"SUMMARY_MAX_ITEMS": "100"}, clear=True):
        self.assertEqual(generate_summaries.get_summary_item_limit(), 100)

def test_summary_workers_never_exceed_five_for_one_hundred_tasks(self):
    with patch.dict(os.environ, {"SUMMARY_WORKERS": "100"}, clear=True):
        self.assertEqual(generate_summaries.get_summary_worker_count(100), 5)
```

Extend the existing batch test to create 105 pending entries and assert exactly the first 100 are submitted and written.

- [ ] **Step 2: Run the DeepSeek tests and confirm the old five-item cap fails**

Run: `py -3.13 -m unittest tests.test_deepseek_config -v`

Expected: failure because `get_summary_item_limit()` is missing and the current batch slices at five.

- [ ] **Step 3: Implement independent item and worker limits**

```python
MAX_SUMMARIES_PER_RUN = 100
MAX_SUMMARY_WORKERS = 5
DEFAULT_SUMMARY_WORKERS = 5

def get_summary_item_limit() -> int:
    configured = _get_positive_int_env("SUMMARY_MAX_ITEMS", MAX_SUMMARIES_PER_RUN)
    return min(configured, MAX_SUMMARIES_PER_RUN)

def get_summary_worker_count(task_count: int) -> int:
    configured = _get_positive_int_env("SUMMARY_WORKERS", DEFAULT_SUMMARY_WORKERS)
    return min(configured, task_count, MAX_SUMMARY_WORKERS)
```

Replace `pending_entries[:MAX_SUMMARIES_PER_RUN]` with `pending_entries[:get_summary_item_limit()]`. Preserve immediate `papers.md` writes after each completed future so interrupted runs skip completed summaries.

- [ ] **Step 4: Run tests and commit the summary change**

Run: `py -3.13 -m unittest tests.test_deepseek_config -v`

Expected: all DeepSeek tests pass; 100 entries are eligible while concurrent futures never exceed five.

```powershell
git add scripts/generate_summaries.py tests/test_deepseek_config.py
git commit -m "feat: summarize up to one hundred papers incrementally"
```

---

### Task 3: Update GitHub workflow, image capacity, and operator documentation

**Files:**
- Modify: `.github/workflows/deploy.yml`
- Modify: `README.md`
- Test: `tests/test_arxiv_crawler.py`
- Test: `tests/test_deepseek_config.py`

**Interfaces:**
- Consumes: Task 1 environment variables and Task 2 `SUMMARY_MAX_ITEMS`.
- Produces: scheduled daily-20 configuration, 100-image processing capacity, and accurate README commands.

- [ ] **Step 1: Add failing workflow assertions**

```python
self.assertIn("ARXIV_INIT_TOTAL_LIMIT: 100", workflow)
self.assertIn("ARXIV_DAILY_TOTAL_LIMIT: 20", workflow)
self.assertIn("ARXIV_DAILY_RESULTS: 20", workflow)
self.assertIn("SUMMARY_MAX_ITEMS: 100", workflow)
self.assertIn("fetch_paper_images.py --max-items 100", workflow)
self.assertIn("build_paper_image_fallback_queue.py --max-items 100", workflow)
```

- [ ] **Step 2: Run workflow-related tests and confirm old values fail**

Run: `py -3.13 -m unittest tests.test_arxiv_crawler tests.test_deepseek_config -v`

Expected: failure on the current daily limit 5 and absent initialization/summary settings.

- [ ] **Step 3: Update workflow values without changing event semantics**

```yaml
env:
  ARXIV_INIT_RESULTS: 80
  ARXIV_INIT_TOTAL_LIMIT: 100
  ARXIV_DAILY_RESULTS: 20
  ARXIV_DAILY_TOTAL_LIMIT: 20
  ARXIV_PAGE_SIZE: 20
```

Set `SUMMARY_MAX_ITEMS: 100`, retain `SUMMARY_WORKERS: 5`, and change both image commands to `--max-items 100`. Keep push/default manual crawler calls on `--dry-run`; only schedule or manual `add_new_papers=true` writes the daily batch.

- [ ] **Step 4: Rewrite README limits and commands**

Document `ARXIV_INIT_TOTAL_LIMIT=100`, `ARXIV_DAILY_TOTAL_LIMIT=20`, `SUMMARY_MAX_ITEMS=100`, five workers, `--backfill-to 100`, and image/fallback limits of 100. State that the initial backfill adds only the difference between the current library size and 100.

- [ ] **Step 5: Run full tests, parse YAML, and commit**

Run: `py -3.13 -m unittest discover -s tests -v`

Expected: all tests pass.

Run: `py -3.13 -c "import yaml; yaml.safe_load(open(r'.github/workflows/deploy.yml', encoding='utf-8')); print('yaml_parse=ok')"`

Expected: `yaml_parse=ok`.

```powershell
git add .github/workflows/deploy.yml README.md tests/test_arxiv_crawler.py tests/test_deepseek_config.py
git commit -m "ci: configure 100-paper initialization and daily twenty"
```

---

### Task 4: Execute the one-time backfill and verify restart safety

**Files:**
- Modify: `papers.md`
- Modify: `site/assets/data.json`
- Modify: `site/assets/paper-images.json`
- Create: `site/assets/paper-images/*.webp`
- Create: `site/assets/paper-images/thumbs/*.webp`
- Modify/Create: generated `site/papers/*/index.html` and `site/covers/*/index.html`

**Interfaces:**
- Consumes: Task 1 `--backfill-to 100`, Task 2 incremental summaries, and Task 3 image capacity.
- Produces: a committed 100-paper static site and public deployment evidence.

- [ ] **Step 1: Record the five-paper baseline without revealing secrets**

```powershell
$beforeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath 'papers.md').Hash
$beforeCount = (Get-Content -LiteralPath 'papers.md' | Where-Object { $_ -match '^\|\s*20\d{2}-\d{2}-\d{2}\s*\|' } | Measure-Object).Count
"before_count=$beforeCount"
"before_sha256=$beforeHash"
```

Expected: `before_count=5`.

- [ ] **Step 2: Backfill exactly to 100 and verify the second run is a no-op**

Run: `py -3.13 -u scripts/arxiv_crawler.py --backfill-to 100`

Expected: 95 new rows and 100 total rows.

Run the same command again.

Expected: zero new rows and a message that the library already has 100 papers.

- [ ] **Step 3: Generate all pending DeepSeek summaries with checkpointing**

```powershell
$env:SUMMARY_MAX_ITEMS='100'
$env:SUMMARY_WORKERS='5'
py -3.13 -u scripts/generate_summaries.py
```

Expected: up to 95 pending summaries complete with no more than five simultaneous requests. A rerun processes only any retained “待生成” entries.

- [ ] **Step 4: Fetch images and run browser fallback for missing entries**

```powershell
py -3.13 -u scripts/fetch_paper_images.py --max-items 100
py -3.13 -u scripts/build_paper_image_fallback_queue.py --max-items 100
npm install
npx playwright install chromium
npm run paper-image:fallbacks
py -3.13 -u scripts/register_paper_image_fallbacks.py
```

Expected: direct images are saved incrementally, remaining items receive Playwright screenshots where figures exist, and reruns skip usable manifest entries.

- [ ] **Step 5: Build and verify the local site**

Run: `py -3.13 -u scripts/build_site.py`

Expected: `site/assets/data.json` contains 100 records; WebP originals and thumbnails are produced for every successfully acquired image.

Use the local browser at `http://127.0.0.1:8000/` and verify:
- the header reports 100 papers;
- 100 `.feed-card` elements exist;
- loaded `.paper-figure-image` elements have nonzero natural dimensions;
- a representative detail page shows its full-resolution image and Chinese summary;
- browser console contains no errors.

- [ ] **Step 6: Run final integrity checks and commit generated content**

Run: `py -3.13 -m unittest discover -s tests -v`

Expected: all tests pass.

Run: `git diff --check` and `git grep -n -E 'sk-[A-Za-z0-9]+' -- ':!*.lock'`.

Expected: no whitespace errors and no tracked DeepSeek key.

```powershell
git add papers.md
git add -f site
git commit -m "data: backfill autonomous-driving library to one hundred papers"
```

- [ ] **Step 7: Push, monitor GitHub Actions, and verify public assets**

Run: `git push origin master`.

Monitor the new Actions run until build and deploy both conclude `success`. Confirm the arXiv preview, DeepSeek check, image fetch, fallback, build, artifact upload, and Pages deployment steps all run without `cancelled` or `skipped` conclusions.

Verify `https://shihongyuan.cn/daily-arxiv-vla/assets/data.json` contains 100 records and sample image URLs return HTTP 200. Open `https://shihongyuan.cn/daily-arxiv-vla/` in the browser and confirm the visible count, card rendering, images, and detail page.

---

## Self-Review Result

- Spec coverage: exact 5-to-100 backfill, persistent accumulation, daily 20, DeepSeek checkpointing, image checkpointing, workflow semantics, tests, local visual validation, and public deployment validation are all assigned to tasks.
- Placeholder scan: no unfinished implementation markers or unspecified error-handling steps remain.
- Type consistency: `backfill_to(target_total: int) -> int`, `get_summary_item_limit() -> int`, environment names, CLI names, and workflow values are consistent across tasks.
