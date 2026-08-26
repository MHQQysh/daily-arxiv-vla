import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
MAX_SUMMARIES_PER_RUN = 100
MAX_SUMMARY_WORKERS = 5
DEFAULT_SUMMARY_WORKERS = 5

AUTONOMOUS_DRIVING_SUMMARY_PROMPT = """你是一名自动驾驶论文阅读专家。只能根据提供的 arXiv 论文 HTML 原文生成中文结构化总结，不得补造论文没有报告的实验信息。

严格使用 Markdown 二级标题和无序列表，输出以下六个部分：

## 研究单位
- 每个机构单独一项，严格写成“机构：机构全称；作者：与该机构对应的作者姓名”。
- 不得只写机构名；如果原文无法确定作者与机构的对应关系，作者部分明确写“原文未明确对应”。

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


"""
/**
 * @file generate_summaries.py
 * @description 读取项目根目录 `papers.md`，为“简要总结”列仍为“待生成”的条目生成摘要，
 * 使用 DeepSeek 官方 OpenAI 兼容 API，并将结果回写到 `papers.md`。
 */
"""


def get_client() -> OpenAI:
    """
    构造直连 DeepSeek 官方 API 的 OpenAI 兼容客户端。
    """
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")

    base_url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL).strip()
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model() -> str:
    """返回单一 DeepSeek 摘要模型，允许通过环境变量覆盖。"""
    return (os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL).strip()


def _get_positive_int_env(name: str, default: int) -> int:
    """读取正整数环境变量，避免无效并发数或写入批大小静默生效。"""
    raw_value = (os.getenv(name) or str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正整数") from exc
    if value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return value


def get_summary_item_limit() -> int:
    """返回本轮最多处理条目数，硬上限为 100。"""
    configured_limit = _get_positive_int_env(
        "SUMMARY_MAX_ITEMS",
        MAX_SUMMARIES_PER_RUN,
    )
    return min(configured_limit, MAX_SUMMARIES_PER_RUN)


def get_summary_workers(task_count: int) -> int:
    """返回本轮实际并发数；无论任务量多大都不超过 5。"""
    if task_count < 1:
        return 0
    configured_workers = _get_positive_int_env(
        "SUMMARY_WORKERS",
        DEFAULT_SUMMARY_WORKERS,
    )
    return min(configured_workers, task_count, MAX_SUMMARY_WORKERS)


def _safe_error_repr(exc: Exception) -> str:
    """保留错误诊断信息，同时避免环境中的 API 密钥进入日志。"""
    message = repr(exc)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message


def get_papers_md_path() -> str:
    """
    /**
     * @function get_papers_md_path
     * @description 获取项目根目录下的 `papers.md` 绝对路径。
     */
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(scripts_dir)
    return os.path.join(root_dir, "papers.md")


def is_placeholder_summary(cell: str) -> bool:
    """
    /**
     * @function is_placeholder_summary
     * @description 判断“简要总结”单元格是否为默认占位（待生成）。
     */
    """
    return "待生成" in cell


def parse_table_line(line: str) -> List[str]:
    """
    /**
     * @function parse_table_line
     * @description 解析 markdown 表格行，返回去除空项后的单元格列表。
     * @param {str} line - 形如 `| a | b | c | d |\n`
     * @returns {List[str]} 单元格列表
     */
    """
    parts = [p.strip() for p in line.strip().split("|")]
    # 去除首尾空项（因为行首尾都有 `|`）
    cells = [p for p in parts if p and p != "---"]
    return cells


def rebuild_line(date_str: str, title: str, link: str, summary_html: str) -> str:
    """
    /**
     * @function rebuild_line
     * @description 将四列内容重建为表格行。
     */
    """
    safe_title = title.replace("|", "\\|")
    safe_summary = summary_html.replace("|", "\\|")
    return f"| {date_str} | {safe_title} | {link} | {safe_summary} |\n"


def generate_summary_for_link(client: OpenAI, link: str, model: str = None) -> str:
    """
    抓取 arXiv HTML 原文并让模型基于 HTML 生成简要总结。
    HTML 和 API 分别重试；API 始终使用同一个 DeepSeek 模型。
    """
    current_model = model or get_model()

    # 将 /abs/ 链接转换为 /html/ 页面
    html_url = re.sub(r"/abs/", "/html/", link)

    # 抓取 HTML 文本（带重试）
    max_retries = int(os.getenv("HTTP_MAX_RETRIES", "3"))
    timeout = int(os.getenv("HTTP_TIMEOUT", "30"))
    html_content = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(html_url, timeout=timeout)
            resp.raise_for_status()
            html_content = resp.text
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"警告: HTML页面不存在，尝试使用PDF: {link}")
                # 如果HTML不存在，尝试获取摘要（fallback）
                return ""
            elif attempt < max_retries - 1:
                print(f"HTTP错误 {e.response.status_code}，重试 {attempt + 1}/{max_retries}: {link}")
                time.sleep(2 ** attempt)
            else:
                print(f"HTTP请求失败，已达最大重试次数: {link}")
                return ""
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"网络错误，重试 {attempt + 1}/{max_retries}: {link}")
                time.sleep(2 ** attempt)
            else:
                print(f"网络请求失败: {link}: {repr(e)}")
                return ""

    if not html_content:
        return ""

    # 按需截断，避免上下文过长
    max_chars = int(os.getenv("HTML_MAX_CHARS", "180000"))
    if len(html_content) > max_chars:
        html_content = html_content[:max_chars]

    api_max_retries = _get_positive_int_env("API_MAX_RETRIES", 3)
    print(f"使用模型生成摘要: {current_model}")

    for attempt in range(api_max_retries):
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=[
                    {
                        "role": "system",
                        "content": AUTONOMOUS_DRIVING_SUMMARY_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"以下为论文的 HTML 原文（可能已截断）：\n\n{html_content}",
                    },
                ],
                stream=False,
                max_tokens=_get_positive_int_env("SUMMARY_MAX_TOKENS", 2048),
                extra_body={"thinking": {"type": "disabled"}},
            )

            if not response.choices:
                raise RuntimeError("API 返回无 choices")

            text = getattr(response.choices[0].message, "content", "") or ""
            if not text.strip():
                raise RuntimeError("API 返回 content 为空")

            text = text.strip()
            # 移除模型可能输出的 <think>...</think> 思考内容
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            # 移除 Markdown 代码块标记
            text = re.sub(r"```markdown\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
            text = text.strip()
            # 规范化换行：保留换行符，但规范化空白
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" +\n", "\n", text)
            # 将换行符转换为 <br> 标签以便在 Markdown 表格中存储
            text = text.replace("\n", "<br>")

            if not text:
                raise RuntimeError("处理后文本为空")

            print(f"✓ 使用模型 {current_model} 成功生成摘要")
            return text
        except Exception as exc:
            safe_error = _safe_error_repr(exc)
            if attempt < api_max_retries - 1:
                print(f"API 调用失败，重试 {attempt + 1}/{api_max_retries}: {safe_error}")
                time.sleep(2 ** attempt)
            else:
                print(f"✗ 模型 {current_model} 已达最大重试次数: {safe_error}")

    print(f"✗ 模型 {current_model} 无法生成摘要: {link}")
    return ""


def default_summary_cell() -> str:
    """
    /**
     * @function default_summary_cell
     * @description 默认折叠占位单元格 HTML。
     */
    """
    return "<details><summary>展开</summary>待生成</details>"


def wrap_in_details(summary_text: str) -> str:
    """
    /**
     * @function wrap_in_details
     * @description 将纯文本包装为折叠 HTML。
     */
    """
    return f"<details><summary>展开</summary>{summary_text}</details>"


def update_papers_md() -> Tuple[int, int]:
    """
    /**
     * @function update_papers_md
     * @description 读取 `papers.md`，为缺失摘要的条目生成并写回。
     * @returns {Tuple[int,int]} (本轮选中数, 实际更新成功数)
     */
    """
    papers_md = get_papers_md_path()
    if not os.path.exists(papers_md):
        raise FileNotFoundError(f"未找到 {papers_md}，请先运行爬取初始化")

    with open(papers_md, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return 0, 0

    header = lines[:2]
    body = lines[2:]

    pending_entries: List[Tuple[int, str, str, str]] = []
    for idx, line in enumerate(body):
        if not line.strip().startswith("|"):
            continue
        cells = parse_table_line(line)
        if len(cells) != 4:
            continue
        date_str, title, link, summary_cell = cells
        if not is_placeholder_summary(summary_cell):
            continue
        pending_entries.append((idx, date_str, title, link))

    entries_to_update = pending_entries[:get_summary_item_limit()]
    need_count = len(entries_to_update)
    if need_count == 0:
        return 0, 0

    success_count = 0
    batch_size = _get_positive_int_env("BATCH_WRITE_SIZE", 5)
    updates_since_last_write = 0
    workers = get_summary_workers(need_count)
    client = get_client()

    remaining_count = len(pending_entries) - need_count
    print(
        f"本轮摘要: {need_count} 篇，workers: {workers}，"
        f"本轮后仍待处理: {remaining_count} 篇"
    )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deepseek-summary") as executor:
        future_entries = {
            executor.submit(generate_summary_for_link, client, entry[3]): entry
            for entry in entries_to_update
        }
        progress_bar = tqdm(
            as_completed(future_entries),
            total=need_count,
            desc="生成简要总结",
            unit="篇",
        )

        for future in progress_bar:
            idx, date_str, title, link = future_entries[future]
            try:
                summary_text = future.result()
                if not summary_text:
                    print(f"警告: 生成摘要为空，跳过: {link}")
                    continue
                new_summary_cell = wrap_in_details(summary_text)
                body[idx] = rebuild_line(date_str, title, link, new_summary_cell)
                success_count += 1
                updates_since_last_write += 1
                progress_bar.set_postfix({"成功": success_count})

                if updates_since_last_write >= batch_size:
                    with open(papers_md, "w", encoding="utf-8") as f:
                        f.writelines(header + body)
                    updates_since_last_write = 0
            except Exception as exc:
                print(f"生成摘要失败: {link}: {_safe_error_repr(exc)}")

    # 最后写入一次，确保所有更改都保存
    if updates_since_last_write > 0:
        try:
            with open(papers_md, "w", encoding="utf-8") as f:
                f.writelines(header + body)
        except Exception as e:
            print(f"错误: 最终写入文件失败: {repr(e)}")
            raise

    return need_count, success_count


if __name__ == "__main__":
    total, updated = update_papers_md()
    print(f"需要生成摘要的条目: {total}，已更新: {updated}")
