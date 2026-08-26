# 每日自动驾驶论文站

这是一个自动收集、中文精读并展示自动驾驶 arXiv 论文的静态网站。页面保留统一的每日论文流，后台分别检索多个技术方向，在写入前按规范化 arXiv ID 合并去重。

## 覆盖方向

- 综合自动驾驶
- 感知：BEV、三维检测、跟踪、车道、占用预测与多传感器融合
- 定位与建图：SLAM、里程计和高精地图
- 运动与行为预测
- 规划与决策
- 车辆控制
- 端到端驾驶、基础模型与世界模型

## 功能特性

- 每日分方向抓取 arXiv 最新论文，跨方向去重并按发布日期排序，单次全站最多新增 `5` 篇
- 使用 DeepSeek 官方 API 和 `deepseek-v4-flash` 生成自动驾驶专用中文摘要
- 从论文 HTML 提取首图，无法直接下载时用 Playwright 截取首个 figure 作为兜底
- 生成响应式首页和独立详情页，支持标题、机构、摘要和 arXiv ID 搜索
- 按 arXiv ID 保存论文详情页阅读进度
- GitHub Actions 每日北京时间 12:00 自动更新并部署 GitHub Pages

## 检索配置

未设置 `ARXIV_QUERY_KEYWORD` 时，爬虫使用 [`scripts/autonomous_driving_topics.py`](scripts/autonomous_driving_topics.py) 中的七组默认查询。每组查询独立执行；单组失败不会中断其他方向，全部失败时任务才会失败。结果经过自动驾驶相关性过滤，跨方向按去除版本号后的 arXiv ID 合并。

- `ARXIV_INIT_RESULTS`：首次初始化时每个方向的最大候选数，默认 `80`
- `ARXIV_INIT_TOTAL_LIMIT`：首次初始化或补齐时全站总量上限，最高 `100`
- `ARXIV_DAILY_RESULTS`：每日更新时每个方向的最大候选数，GitHub 工作流使用 `20`
- `ARXIV_DAILY_TOTAL_LIMIT`：排除已有论文后，每日最终新增硬上限，最高 `20`
- `ARXIV_PAGE_SIZE`：单次 arXiv 请求页大小，默认 `20`
- `ARXIV_DELAY_SECONDS`：arXiv 请求间隔，默认 `10` 秒
- `ARXIV_RETRY_BASE_SECONDS`：失败后的指数退避基数，默认 `30` 秒
- `ARXIV_MAX_RETRIES`：每个方向的最大请求次数，默认 `3`
- `ARXIV_KEYWORD_LABEL`：页面显示名称，默认 `自动驾驶`

设置 `ARXIV_QUERY_KEYWORD` 后只执行这一条自定义 arXiv 查询，不再运行七组默认查询。例如在 PowerShell 中：

```powershell
$env:ARXIV_QUERY_KEYWORD = 'all:"autonomous driving" AND all:"simulation"'
python scripts/arxiv_crawler.py
Remove-Item Env:ARXIV_QUERY_KEYWORD
```

首次运行且 `papers.md` 没有论文记录时最多建立 100 篇论文库；已有少量数据时可执行 `python scripts/arxiv_crawler.py --backfill-to 100`，只补充距离 100 篇所差的数量。后续定时运行会先完成相关性过滤、跨方向去重并排除已有论文，再按发布日期顺序每日新增最多 20 篇，历史论文持续保留。

## DeepSeek 摘要配置

摘要脚本只使用一个 DeepSeek 模型，不进行跨模型或跨供应商回退：

- `DEEPSEEK_API_KEY`：DeepSeek API 密钥
- `DEEPSEEK_BASE_URL`：默认 `https://api.deepseek.com`
- `DEEPSEEK_MODEL`：默认 `deepseek-v4-flash`
- 摘要脚本每次固定最多处理 `5` 篇，失败条目保留“待生成”供下一次补齐
- `SUMMARY_WORKERS`：摘要并发数，默认并最高为 `5`
- `SUMMARY_MAX_ITEMS`：单轮最多处理的待生成摘要数，最高 `100`
- `SUMMARY_MAX_TOKENS`：单篇摘要最大输出 token，默认 `2048`
- `API_MAX_RETRIES`：同一模型的 API 调用次数，默认 `3`
- `HTTP_MAX_RETRIES`：抓取论文 HTML 的最大次数，默认 `3`
- `HTTP_TIMEOUT`：抓取论文 HTML 的超时秒数，默认 `30`
- `HTML_MAX_CHARS`：送入模型的 HTML 最大字符数，默认 `180000`
- `BATCH_WRITE_SIZE`：每成功生成多少篇后写回一次，默认 `5`

项目依赖中的 `openai` Python 包仅作为 OpenAI 兼容协议客户端使用。默认流量路径是“本地或 GitHub Actions → `https://api.deepseek.com` → DeepSeek 官方 API”，不会把请求发送到 OpenAI，也没有其他模型供应商兜底。

复制示例配置后填写密钥即可在本地生成摘要；`.env` 只保存在本机，不应提交：

```powershell
Copy-Item .env.example .env
# 编辑 .env，只填写自己的 DEEPSEEK_API_KEY
python scripts/generate_summaries.py
```

`DEEPSEEK_API_KEY` 只影响摘要生成。没有密钥时仍可抓取论文、提取图片、构建和部署网站，摘要保留“待生成”；GitHub Actions 会安全跳过摘要步骤，之后配置 Secret 再补齐。密钥只应通过本地环境变量、未提交的 `.env` 或 GitHub Secret 传入，脚本和工作流都不会打印它。

## 本地运行

### 1. 安装依赖并运行测试

```powershell
python -m pip install -r requirements.txt
npm install
npx playwright install chromium
python -m unittest discover -s tests -v
```

### 2. 爬取论文

```powershell
python scripts/arxiv_crawler.py
```

脚本根据 `papers.md` 是否已有论文记录自动选择初始化或每日更新模式。

### 3. 生成摘要（可选）

仅在已经配置 `DEEPSEEK_API_KEY` 时运行：

```powershell
python scripts/generate_summaries.py
```

### 4. 抓取论文首图

```powershell
# 直接从 arXiv HTML 提取图片
python scripts/fetch_paper_images.py --max-items 100

# 为仍缺图的论文建立截图兜底队列
python scripts/build_paper_image_fallback_queue.py --max-items 100

# 截取首个 figure，并注册进图片 manifest
npm run paper-image:fallbacks
python scripts/register_paper_image_fallbacks.py
```

### 5. 构建与预览

```powershell
python scripts/build_site.py
python -m http.server 8000 --directory site
```

浏览器打开 `http://127.0.0.1:8000`。构建结果位于 `site/`，包括首页、轻量数据文件、论文图片资源和每篇论文的独立详情页。

## Google Analytics 4

GA4 完全可选。构建前设置 Measurement ID：

```powershell
$env:GA_MEASUREMENT_ID = 'G-XXXXXXXXXX'
python scripts/build_site.py
```

未配置或格式不正确时，页面不会加载 Google Analytics，也不会发送自定义统计事件。可在浏览器开发者工具的 Network 面板搜索 `googletagmanager` 或 `collect` 验证；实时报告通常有几分钟延迟。

Measurement ID 会写入客户端 HTML，不属于密钥。公开部署时建议在 `Settings → Secrets and variables → Actions → Variables` 中添加 `GA_MEASUREMENT_ID`；工作流也兼容同名 Secret。面向公众提供服务时，请根据适用法规补充隐私说明和必要的用户同意机制。

## GitHub Pages 部署

### 仓库设置

1. 在仓库 `Settings → Pages` 中选择 `GitHub Actions` 作为部署源。
2. 如需自动生成摘要，在 `Settings → Secrets and variables → Actions → Secrets` 中添加 `DEEPSEEK_API_KEY`。
3. 如需 GA4，在 Actions Variables 中添加 `GA_MEASUREMENT_ID`。

`DEEPSEEK_API_KEY` 对部署不是必需项。未配置时，工作流只跳过摘要调用，论文抓取、图片处理、站点构建和 Pages 部署继续执行。

### 自动流程

推送到 `master` 或 `main` 会触发工作流；定时任务在每天 UTC 04:00（北京时间 12:00）触发。工作流依次：

1. 安装 Python 和 Playwright 依赖并运行完整单元测试。
2. 按七个方向抓取论文，候选上限为首次每方向 `80`、每日每方向 `10`；过滤、去重并排除已有论文后，全站单次最多写入 `5` 篇。
3. 有 `DEEPSEEK_API_KEY` 时直连 DeepSeek Flash，并发生成本轮最多 `5` 篇摘要；没有时保留“待生成”并继续。
4. 抓取论文首图并执行 Playwright 截图兜底。
5. 注入可选 GA4 配置并运行 `python scripts/build_site.py`。
6. 提交更新后的 `papers.md` 和 `site/`，上传并部署 Pages 产物。

部署成功后，网站地址通常为：

```text
https://你的用户名.github.io/仓库名/
```

## 项目结构

```text
.
├── papers.md
├── scripts/
│   ├── autonomous_driving_topics.py
│   ├── arxiv_crawler.py
│   ├── generate_summaries.py
│   ├── fetch_paper_images.py
│   ├── build_paper_image_fallback_queue.py
│   ├── render_paper_image_fallbacks.mjs
│   ├── register_paper_image_fallbacks.py
│   └── build_site.py
├── tests/
├── site/
│   ├── index.html
│   ├── papers/<arxiv-id>/index.html
│   └── assets/
└── .github/workflows/deploy.yml
```

## 数据格式

`papers.md` 使用固定四列表格：

```markdown
| 日期 | 标题 | 链接 | 简要总结 |
| --- | --- | --- | --- |
| 2026-08-26 | 论文标题 | https://arxiv.org/abs/2608.00001 | <details><summary>展开</summary>待生成</details> |
```

## 技术栈

- Python 3.9
- arXiv Python 库与 Requests
- DeepSeek 官方 OpenAI 兼容 API
- 原生 HTML、CSS 和 JavaScript
- Playwright
- GitHub Actions 与 GitHub Pages
