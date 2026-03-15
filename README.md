# Olist Growth Copilot

一个可在 macOS 本地运行的电商/增长分析对话式数据分析助手 Demo（Streamlit + DuckDB + OpenAI）。

## 功能

- 中文自然语言提问，自动生成 DuckDB SQL
- 优先使用 OpenAI API 生成 SQL；未配置 `OPENAI_API_KEY` 时自动走规则 fallback
- SQL Guard 安全校验：
  - 仅允许 `SELECT` / `WITH`
  - 禁止 `INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE/COPY` 等
  - 仅允许白名单表/视图
  - 若缺少 `LIMIT` 自动追加默认值
- 执行 SQL 并展示 DataFrame
- 基于查询结果给出一句简要洞察
- 支持 GMV 异动归因（最近N天 vs 前N天）：拆解订单量效应/AOV效应，并给出州/品类/商家贡献
- 自动生成老板可读结论（例如“本期 GMV 下滑 xx%，主要由订单量下降驱动，占总降幅 xx%”）
- 侧边栏示例问题按钮可一键填入输入框
- 内置最小评测脚本：统计生成成功率、Guard 通过率、执行成功率

## 项目结构

```text
olist-growth-copilot/
  app.py
  requirements.txt
  .env.example
  README.md

  metrics/
    metrics.yml

  agent/
    __init__.py
    metrics_store.py
    sql_guard.py
    llm_sql.py
    insight.py

  warehouse/
    load_olist_to_duckdb.py
    views.sql

  eval/
    questions.jsonl
    run_eval.py
```

## 运行步骤

1. 创建虚拟环境并安装依赖

```bash
cd olist-growth-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 下载并解压 Olist CSV 到 `./olist_data/`

建议包含以下文件（至少这些）：

- `olist_orders_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_geolocation_dataset.csv`

3. 生成 DuckDB 数据库

```bash
python warehouse/load_olist_to_duckdb.py
```

执行后会在项目根目录生成 `olist.duckdb`，并创建视图：

- `vw_eligible_orders`
- `vw_fact_items`
- `vw_delivered_orders`

4. 配置环境变量（OpenAI Key 可选）

```bash
cp .env.example .env
# 可编辑 .env
# OPENAI_API_KEY=你的key
# OPENAI_MODEL=gpt-4o-mini
# OLIST_DB_PATH=olist.duckdb
```

5. 启动应用

```bash
streamlit run app.py
```

## Eval（最小版）

```bash
python eval/run_eval.py
```

输出示例：

```json
{
  "total": 32,
  "generation_success": 32,
  "guard_pass": 31,
  "execution_success": 28,
  "generation_success_rate_pct": 100.0,
  "guard_pass_rate_pct": 96.88,
  "execution_success_rate_pct": 87.5
}
```

## 说明

- 若未配置 `OPENAI_API_KEY`，应用不会报错，会自动使用规则 fallback 生成 SQL。
- LLM prompt 注入了 `metrics/metrics.yml` 的压缩上下文，约束指标口径（例如 GMV 不计取消订单）。
- SQL Guard 在执行前进行防护，避免写操作和非白名单对象访问。
