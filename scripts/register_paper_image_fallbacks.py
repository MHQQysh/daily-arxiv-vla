#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 Playwright 截图兜底生成的 PNG 注册到论文首图 manifest 中。
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import requests

from fetch_paper_images import MANIFEST_PATH, load_manifest, save_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = PROJECT_ROOT / "tmp" / "paper-image-fallback-queue.json"
RESULT_PATH = PROJECT_ROOT / "tmp" / "paper-image-fallback-results.json"


def main() -> int:
    if not QUEUE_PATH.exists() or not RESULT_PATH.exists():
        print("未找到兜底队列或截图结果，跳过注册。")
        return 0

    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    manifest = load_manifest()

    queue_by_id = {item["arxiv_id"]: item for item in queue}
    updated = 0

    for result in results:
        if not result.get("ok"):
            continue

        arxiv_id = result.get("arxiv_id", "")
        queue_item = queue_by_id.get(arxiv_id)
        if not queue_item:
            continue

        output_path = Path(queue_item["output_path"])
        if not output_path.exists():
            continue

        manifest[arxiv_id] = {
            "title": queue_item["title"],
            "abs_url": queue_item["abs_url"],
            "html_url": result.get("used_url") or queue_item["html_url"],
            "path": queue_item["relative_path"],
            "image_url": result.get("used_url") or queue_item["html_url"],
            "source": f"playwright:{result.get('selector', 'element')}",
            "score": 60,
            "content_type": "image/png",
            "inside_figure": True,
            "width": result.get("width"),
            "height": result.get("height"),
        }
        updated += 1

    # 没有 arXiv HTML 图形页面时，以论文 PDF 首页作为最终首图兜底。
    for result in results:
        if result.get("ok"):
            continue

        arxiv_id = result.get("arxiv_id", "")
        queue_item = queue_by_id.get(arxiv_id)
        if not queue_item:
            continue

        output_path = Path(queue_item["output_path"])
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            document = fitz.open(stream=response.content, filetype="pdf")
            if document.page_count < 1:
                raise RuntimeError("PDF 没有页面")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pixmap = document.load_page(0).get_pixmap(
                matrix=fitz.Matrix(1.5, 1.5),
                alpha=False,
            )
            pixmap.save(output_path)
            document.close()
        except Exception as exc:
            print(f"PDF 首页兜底失败 {arxiv_id}: {exc!r}")
            continue

        manifest[arxiv_id] = {
            "title": queue_item["title"],
            "abs_url": queue_item["abs_url"],
            "html_url": queue_item["html_url"],
            "path": queue_item["relative_path"],
            "image_url": pdf_url,
            "source": "pdf:first-page",
            "score": 50,
            "content_type": "image/png",
            "inside_figure": False,
            "width": pixmap.width,
            "height": pixmap.height,
        }
        updated += 1
        print(f"PDF 首页兜底成功: {arxiv_id}")

    save_manifest(manifest)
    print(f"已注册截图兜底首图: {updated}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
