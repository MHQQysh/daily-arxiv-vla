import argparse
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Set

import arxiv
import requests

from autonomous_driving_topics import (
	ALLOWED_PRIMARY_CATEGORIES,
	get_topic_queries,
	is_relevant_paper,
)


_ARXIV_ID_PATTERN = r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})(?:v\d+)?"
_ARXIV_VALUE_PATTERN = re.compile(
	r"^\s*(?:(?:https?://(?:(?:www|export)\.)?arxiv\.org/(?:abs|pdf)/)|arxiv:)?"
	r"(?P<identifier>" + _ARXIV_ID_PATTERN + r")(?:\.pdf)?/?(?:[?#].*)?\s*$",
	re.IGNORECASE,
)
_ARXIV_ABS_LINK_PATTERN = re.compile(
	r"https?://(?:(?:www|export)\.)?arxiv\.org/abs/" + _ARXIV_ID_PATTERN,
	re.IGNORECASE,
)
_INIT_TOTAL_LIMIT_HARD_MAX = 100
_DAILY_TOTAL_LIMIT_HARD_MAX = 20


class _TimeoutSession(requests.Session):
	"""为 arxiv.py 未设置 timeout 的 HTTP 请求补上硬超时。"""

	def __init__(self, timeout_seconds: float):
		super().__init__()
		self.timeout_seconds = timeout_seconds

	def request(self, method, url, **kwargs):
		kwargs.setdefault("timeout", self.timeout_seconds)
		return super().request(method, url, **kwargs)


class ArxivCollector:
	"""
	/**
	 * @class ArxivCollector
	 * @description 每日分主题获取 arXiv 上的自动驾驶论文，并维护项目根目录下的
	 * `papers.md` 表格（列：日期、标题、链接）。首次运行无数据时执行初始化，之后每日增量并去重；
	 * 初始化最多写入 100 篇，日常更新最多写入 20 篇。
	 * 可通过环境变量 ARXIV_QUERY_KEYWORD 临时覆盖默认的七组自动驾驶检索主题。
	 */
	"""

	_ALLOWED_PRIMARY_CATEGORIES = ALLOWED_PRIMARY_CATEGORIES

	def __init__(self, papers_path: str, init_results: int = None, daily_results: int = None,
				 query_keyword: str = None, topic_queries: Optional[Dict[str, str]] = None,
				 init_total_limit: int = None, daily_total_limit: int = None):
		"""
		初始化 ArxivCollector
		参数可通过环境变量配置：
		- ARXIV_QUERY_KEYWORD: 可选的单条自定义搜索查询
		- ARXIV_INIT_RESULTS: 初始化时每个主题抓取数量（默认 80）
		- ARXIV_DAILY_RESULTS: 每日每个主题抓取数量（默认 10）
		- ARXIV_INIT_TOTAL_LIMIT: 首次初始化全站写入上限（默认 100）
		- ARXIV_DAILY_TOTAL_LIMIT: 每日全站写入上限（默认 20）
		- ARXIV_PAGE_SIZE: 单次请求返回数量（默认 20，避免 arxiv 库默认请求 100 条触发限流）
		- ARXIV_DELAY_SECONDS: arXiv 请求间隔（默认 10 秒）
		"""
		self.papers_path = papers_path
		init_value = init_results if init_results is not None else int(os.getenv("ARXIV_INIT_RESULTS", "80"))
		daily_value = daily_results if daily_results is not None else int(os.getenv("ARXIV_DAILY_RESULTS", "10"))
		init_total_limit_value = (
			init_total_limit
			if init_total_limit is not None
			else int(os.getenv("ARXIV_INIT_TOTAL_LIMIT", str(_INIT_TOTAL_LIMIT_HARD_MAX)))
		)
		daily_total_limit_value = (
			daily_total_limit
			if daily_total_limit is not None
			else int(os.getenv("ARXIV_DAILY_TOTAL_LIMIT", str(_DAILY_TOTAL_LIMIT_HARD_MAX)))
		)
		self.init_results = self._require_positive_int("init_results", init_value)
		self.daily_results = self._require_positive_int("daily_results", daily_value)
		self.init_total_limit = self._require_positive_int(
			"init_total_limit",
			init_total_limit_value,
			maximum=_INIT_TOTAL_LIMIT_HARD_MAX,
		)
		self.daily_total_limit = self._require_positive_int(
			"daily_total_limit",
			daily_total_limit_value,
			maximum=_DAILY_TOTAL_LIMIT_HARD_MAX,
		)
		override = query_keyword or os.getenv("ARXIV_QUERY_KEYWORD")
		self.topic_queries = topic_queries if topic_queries is not None else get_topic_queries(override)
		self.arxiv_page_size = self._require_positive_int(
			"ARXIV_PAGE_SIZE",
			int(os.getenv("ARXIV_PAGE_SIZE", "20")),
			maximum=2000,
		)
		self.arxiv_max_retries = self._require_positive_int(
			"ARXIV_MAX_RETRIES",
			int(os.getenv("ARXIV_MAX_RETRIES", "3")),
		)
		self.arxiv_delay_seconds = float(os.getenv("ARXIV_DELAY_SECONDS", "10"))
		self.arxiv_request_timeout_seconds = self._require_positive_float(
			"ARXIV_REQUEST_TIMEOUT_SECONDS",
			float(os.getenv("ARXIV_REQUEST_TIMEOUT_SECONDS", "15")),
		)
		self._client = arxiv.Client(
			page_size=self.arxiv_page_size,
			delay_seconds=self.arxiv_delay_seconds,
			num_retries=0,
		)
		self._client._session = _TimeoutSession(self.arxiv_request_timeout_seconds)

	@staticmethod
	def _require_positive_int(name: str, value: int, maximum: Optional[int] = None) -> int:
		if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
			raise ValueError(f"{name} 必须是正整数")
		if maximum is not None and value > maximum:
			raise ValueError(f"{name} 不能大于 {maximum}")
		return value

	@staticmethod
	def _require_positive_float(name: str, value: float) -> float:
		if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
			raise ValueError(f"{name} 必须是正数")
		return float(value)

	def _search(self, query: str, max_results: int) -> List[arxiv.Result]:
		"""
		搜索 arXiv 论文，带重试机制
		"""
		self._require_positive_int("max_results", max_results)
		retry_base_seconds = float(os.getenv("ARXIV_RETRY_BASE_SECONDS", "30"))

		last_error = None
		for attempt in range(self.arxiv_max_retries):
			try:
				print(
					f"arXiv 请求 {attempt + 1}/{self.arxiv_max_retries}，"
					f"最多返回 {max_results} 篇",
					flush=True,
				)
				search = arxiv.Search(
					query=query,
					max_results=max_results,
					sort_by=arxiv.SortCriterion.SubmittedDate,
					sort_order=arxiv.SortOrder.Descending,
				)
				results = list(self._client.results(search))
				print(f"arXiv 请求完成，返回 {len(results)} 篇", flush=True)
				return results
			except Exception as exc:
				last_error = exc
				if attempt < self.arxiv_max_retries - 1:
					wait_seconds = retry_base_seconds * (2 ** attempt)
					print(
						f"arXiv 搜索失败，{wait_seconds:.0f} 秒后重试 "
						f"{attempt + 1}/{self.arxiv_max_retries}: {exc!r}",
						flush=True,
					)
					time.sleep(wait_seconds)
		raise RuntimeError(f"arXiv 搜索达到最大重试次数: {last_error!r}")

	def _collect(self, max_results: int) -> List[arxiv.Result]:
		self._require_positive_int("max_results", max_results)
		merged: Dict[str, arxiv.Result] = {}
		failed_topics = []
		for topic_name, query in self.topic_queries.items():
			print(f"检索主题: {topic_name}", flush=True)
			try:
				results = self._search(query, max_results)
			except Exception as exc:
				failed_topics.append(topic_name)
				print(f"主题检索失败，继续其他主题: {topic_name}: {exc!r}", flush=True)
				continue
			for result in results:
				if result.primary_category not in self._ALLOWED_PRIMARY_CATEGORIES:
					continue
				if not is_relevant_paper(result.title or "", result.summary or ""):
					continue
				canonical_id = self._canonical_arxiv_id(result.entry_id)
				if canonical_id is None:
					print(f"跳过无法识别的 arXiv 链接: {result.entry_id!r}")
					continue
				current = merged.get(canonical_id)
				if current is None or self._revision_time(result) > self._revision_time(current):
					merged[canonical_id] = result
		if len(failed_topics) == len(self.topic_queries):
			raise RuntimeError(f"全部 {len(self.topic_queries)} 个自动驾驶主题检索失败")
		ordered = [merged[canonical_id] for canonical_id in sorted(merged)]
		return sorted(ordered, key=lambda result: result.published, reverse=True)

	@staticmethod
	def _revision_time(result: arxiv.Result) -> datetime:
		return getattr(result, "updated", None) or result.published

	@staticmethod
	def _canonical_arxiv_id(value: str) -> Optional[str]:
		match = _ARXIV_VALUE_PATTERN.fullmatch(value or "")
		if match is None:
			return None
		identifier = re.sub(r"v\d+$", "", match.group("identifier"), flags=re.IGNORECASE)
		return identifier.lower()

	def has_existing_papers(self) -> bool:
		return bool(self._load_existing_links())

	def _normalize_link(self, link: str) -> str:
		"""
		/**
		 * @private 将新式或旧式 arXiv 链接/ID 规范化为 HTTPS abs 链接并去掉版本号。
		 * @param {str} link - 原始链接或 arXiv ID
		 * @returns {str} 规范化后的链接（无版本号）
		 */
		"""
		identifier = self._canonical_arxiv_id(link)
		if identifier is None:
			return link.strip()
		return f"https://arxiv.org/abs/{identifier}"

	def _default_summary_cell(self) -> str:
		"""
		/**
		 * @private 返回简要总结列的默认折叠占位。
		 */
		"""
		return "<details><summary>展开</summary>待生成</details>"

	def _ensure_md_header(self) -> None:
		"""
		/**
		 * 确保 `papers.md` 存在且包含四列表头。
		 */
		"""
		four_header = "| 日期 | 标题 | 链接 | 简要总结 |\n"
		four_sep = "| --- | --- | --- | --- |\n"
		if not os.path.exists(self.papers_path):
			with open(self.papers_path, "w", encoding="utf-8") as f:
				f.write(four_header)
				f.write(four_sep)

	def _load_existing_links(self) -> Set[str]:
		"""
		解析 papers.md 已有的 arXiv 链接，用规范化后的 arXiv ID 集合去重。
		"""
		if not os.path.exists(self.papers_path):
			return set()

		identifiers: Set[str] = set()
		try:
			with open(self.papers_path, "r", encoding="utf-8") as f:
				for line in f:
					for m in _ARXIV_ABS_LINK_PATTERN.findall(line):
						canonical_id = self._canonical_arxiv_id(m)
						if canonical_id is not None:
							identifiers.add(canonical_id)
		except Exception as e:
			print(f"警告: 读取 papers.md 失败: {repr(e)}")
			return set()

		return identifiers

	def _format_row(self, r: arxiv.Result) -> str:
		"""
		/**
		 * 将单条结果格式化为 Markdown 表格行（四列）。
		 * @param {Result} r - 论文结果
		 * @returns {str} 形如 `| 2025-09-26 | 标题 | https://arxiv.org/abs/xxxx | <details>..</details> |`
		 */
		"""
		date_str = r.published.strftime("%Y-%m-%d") if isinstance(r.published, datetime) else ""
		title = (r.title or "").replace("|", "\\|").strip()
		# 规范化链接，去掉版本号
		link = self._normalize_link(r.entry_id)
		summary_cell = self._default_summary_cell()
		return f"| {date_str} | {title} | {link} | {summary_cell} |\n"

	def _append_rows(self, rows: List[str]) -> None:
		"""
		将若干行插入到表头之后（保持最新内容靠前）。
		"""
		self._ensure_md_header()

		try:
			with open(self.papers_path, "r", encoding="utf-8") as f:
				lines = f.readlines()
		except Exception as e:
			print(f"错误: 读取 papers.md 失败: {repr(e)}")
			raise

		insert_idx = 2 if len(lines) >= 2 else len(lines)
		new_lines = lines[:insert_idx] + rows + lines[insert_idx:]

		try:
			with open(self.papers_path, "w", encoding="utf-8") as f:
				f.writelines(new_lines)
		except Exception as e:
			print(f"错误: 写入 papers.md 失败: {repr(e)}")
			raise

	def _build_new_rows(self, results: List[arxiv.Result], existing: Set[str], limit: int) -> List[str]:
		"""排除已有论文后，按结果顺序生成不超过指定上限的新行。"""
		self._require_positive_int("limit", limit, maximum=_INIT_TOTAL_LIMIT_HARD_MAX)
		rows: List[str] = []
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

	def initialize(self) -> int:
		"""
		/**
		 * 初始化 `papers.md`：抓取较多历史候选，去重后按全站硬上限写入。
		 * @returns {int} 写入的论文数量
		 */
		"""
		self._ensure_md_header()
		existing = self._load_existing_links()
		results = self._collect(self.init_results)
		rows = self._build_new_rows(results, existing, self.init_total_limit)
		if rows:
			self._append_rows(rows)
		return len(rows)

	def run_daily(self) -> int:
		"""
		/**
		 * 每日增量：抓取少量最新论文，与已存在内容去重并按全站硬上限插入表头之后。
		 * @returns {int} 新增的论文数量
		 */
		"""
		self._ensure_md_header()
		existing = self._load_existing_links()
		results = self._collect(self.daily_results)
		rows = self._build_new_rows(results, existing, self.daily_total_limit)
		if rows:
			self._append_rows(rows)
		return len(rows)

	def preview_daily(self) -> int:
		"""执行真实检索与去重，但不修改 papers.md。用于手动验证工作流。"""
		existing = self._load_existing_links()
		results = self._collect(self.daily_results)
		rows = self._build_new_rows(results, existing, self.daily_total_limit)
		print(f"预览完成：可新增 {len(rows)} 篇；未修改 {self.papers_path}", flush=True)
		return len(rows)

	def backfill_to(self, target_total: int) -> int:
		"""补齐到指定总量；已有论文计入目标，重复执行不会超过目标。"""
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
		rows = self._build_new_rows(results, existing, remaining)
		if rows:
			self._append_rows(rows)
		print(f"补齐完成：新增 {len(rows)} 篇，总目标 {target} 篇", flush=True)
		return len(rows)


def _default_papers_path() -> str:
	"""
	/**
	 * 计算项目根目录下的 `papers.md` 绝对路径。
	 * 假设当前文件位于 `<root>/scripts/`。
	 */
	"""
	scripts_dir = os.path.dirname(os.path.abspath(__file__))
	root_dir = os.path.dirname(scripts_dir)
	return os.path.join(root_dir, "papers.md")


def main() -> int:
	parser = argparse.ArgumentParser(description="抓取每日自动驾驶 arXiv 论文")
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="执行真实检索和去重，但不修改 papers.md",
	)
	parser.add_argument(
		"--backfill-to",
		type=int,
		help="将论文库补齐到指定总量（最高 100）",
	)
	args = parser.parse_args()

	papers_md = _default_papers_path()
	collector = ArxivCollector(papers_md)

	if args.backfill_to is not None:
		collector.backfill_to(args.backfill_to)
		return 0

	if args.dry_run:
		collector.preview_daily()
		return 0

	if collector.has_existing_papers():
		count = collector.run_daily()
		print(f"每日更新完成，新增 {count} 篇论文，写入 {papers_md}")
	else:
		count = collector.initialize()
		print(f"初始化完成，新增 {count} 篇论文，写入 {papers_md}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
