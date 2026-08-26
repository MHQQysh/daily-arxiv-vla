import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

from scripts.generate_summaries import (
    AUTONOMOUS_DRIVING_SUMMARY_PROMPT,
    DEEPSEEK_DEFAULT_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
)


def main() -> None:
    """对 DeepSeek 官方兼容接口执行一次手动连通性检查。"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY，未发起网络请求")

    base_url = (os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_DEFAULT_BASE_URL).strip()
    model = (os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL).strip()
    client = OpenAI(api_key=api_key, base_url=base_url)

    url = "https://arxiv.org/html/2509.21243v1"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    html_content = response.text[:180000]

    stream = client.chat.completions.create(
        model=model,
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
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(delta.content, end="", flush=True)


if __name__ == "__main__":
    main()
