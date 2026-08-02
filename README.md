# 电商增长数据分析助手

> 一个可在本地运行的电商增长分析助手：用中文提问，自动生成 SQL、查询数据库、展示图表与结论。

**技术栈**：Streamlit · DuckDB · DeepSeek / OpenAI · Python 3.11

---

## 它能做什么？

你在网页输入一个中文问题，助手完成以下全部步骤：

```text
你的问题（中文）
    ↓
LLM 翻译成 SQL（DeepSeek 或 OpenAI）
    ↓
SQL 安全检查（只读、白名单、自动加 LIMIT）
    ↓
查询本地 DuckDB 数据库
    ↓
展示表格 + 图表 + 一句话摘要 + GMV 归因分析
```

示例问题：

- 近 30 天 GMV 走势（按天）
- 上个月订单数、GMV、客单价分别是多少？
- 按州看 GMV Top 10（近 90 天）
- 准时送达率按月趋势（最近 6 个月）
- 延迟送达订单平均评分 vs 准时订单平均评分
- 新客占比按月趋势（今年以来）

---

## 核心功能

| 功能 | 说明 |
|------|------|
| 中文自然语言查询 | 直接输入中文问题，LLM 自动生成对应 SQL |
| 双 LLM 支持 | 默认使用 DeepSeek V3（deepseek-chat），可切换至 OpenAI |
| 无 Key 降级 | 未配置 API Key 时自动走内置规则 fallback，不报错 |
| SQL 安全防护 | 只允许 SELECT/WITH，禁止写操作，表名白名单，自动补 LIMIT |
| 指标语义约束 | 28 个业务指标定义注入 LLM Prompt，保证口径一致（如 GMV 不含取消订单） |
| 可视化图表 | 支持折线/柱状/面积/散点图，自动识别时间轴 |
| GMV 归因分析 | 自动对比当期 vs 前期，拆解订单量效应/AOV 效应，按州/品类/商家定位贡献 |
| 评测脚本 | 内置 eval 脚本，输出 SQL 生成成功率、Guard 通过率、执行成功率 |

---

## 项目结构

```text
growth-analysis-agent/
├── app.py                        # 网页入口（Streamlit），串联所有模块
├── requirements.txt
├── README.md
│
├── agent/                        # 核心 Agent 逻辑
│   ├── llm_sql.py                # 中文问题 → SQL（LLM 调用 + fallback 规则）
│   ├── sql_guard.py              # SQL 安全检查（只读白名单，自动补 LIMIT）
│   ├── metrics_store.py          # 读取指标词典，压缩成 LLM Prompt
│   ├── insight.py                # 查询结果 → 一句话摘要
│   └── attribution.py            # GMV 异动归因分析
│
├── metrics/
│   └── metrics.yml               # 28 个业务指标定义（中文名/公式/SQL 提示）
│
├── warehouse/
│   ├── load_olist_to_duckdb.py   # 一次性：CSV → DuckDB 数据库初始化
│   └── views.sql                 # 3 个预处理视图定义
│
├── eval/
│   ├── questions.jsonl            # 评测问题集
│   └── run_eval.py               # 评测脚本
│
└── scripts/
    ├── set_deepseek_keychain.sh       # 把 DeepSeek API Key 写入 macOS Keychain
    └── delete_deepseek_keychain.sh
```

### 数据库视图说明

`load_olist_to_duckdb.py` 初始化时会基于原始 CSV 自动创建以下视图，查询时应优先使用它们：

| 视图 | 含义 |
|------|------|
| `vw_eligible_orders` | 有效订单（delivered / shipped / approved / invoiced） |
| `vw_fact_items` | 完整商品明细，含用户、州、GMV、运费等字段 |
| `vw_delivered_orders` | 已妥投订单，用于履约/配送分析 |

---

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备原始数据

将以下 Olist CSV 文件放入项目根目录的 `olist_data/` 文件夹：

```text
olist_data/
├── olist_orders_dataset.csv
├── olist_customers_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── olist_geolocation_dataset.csv
```

> 数据来源：[Kaggle - Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### 3. 初始化数据库（只需运行一次）

```bash
python warehouse/load_olist_to_duckdb.py
```

执行后在项目根目录生成 `olist.duckdb`，并自动创建 3 个分析视图。

### 4. 配置 API Key

#### 方式 A：`.env` 文件（简单）

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
# DeepSeek（默认）
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 或切换到 OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=你的key
# OPENAI_MODEL=gpt-4o-mini
```

#### 方式 B：macOS Keychain（推荐，Key 不落盘）

```bash
./scripts/set_deepseek_keychain.sh
# 按提示输入 DeepSeek API Key（不会回显）
```

应用启动时会自动从 Keychain 读取，无需写入任何文件。

> 若两种方式都未配置，应用会自动降级到内置规则 fallback，仍可正常使用示例问题。

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，即可开始提问。

---

## AI Evals 评测体系

本项目通过 AI Evals 驱动产品质量持续改进，评测脚本覆盖三个层次：

| 层次 | 评测内容 |
| ---- | -------- |
| Layer 1 管道健康度 | SQL 生成成功率 → Guard 通过率 → 执行成功率 |
| Layer 2 语义正确性 | 与 golden SQL 结果对比（行数 / 数值近似） |
| Layer 3 指标合规性 | 静态检查 SQL 是否引用了正确的表和关键模式 |

评测集共 **40 道题**，覆盖 5 个业务类别（GMV / 订单 / 履约 / 评价 / 用户），按 easy / medium / hard 三档难度分层。

### 运行评测

```bash
PYTHONPATH=. python eval/run_eval.py
```

### 最新评测结果（DeepSeek V3 via 硅基流动）

| 指标 | 结果 |
| ---- | ---- |
| SQL 生成成功率 | **100%** |
| Guard 通过率 | **95%** |
| 执行成功率 | **85%** |
| 语义正确率 | **94.12%** |
| 指标合规率 | **100%** |

按难度分布：

| 难度 | 执行成功率 | 语义正确率 |
| ---- | --------- | --------- |
| Easy（14 题） | 78.57% | 90.91% |
| Medium（19 题） | 84.21% | 93.75% |
| Hard（7 题） | 100% | 100% |

### Evals 驱动改进记录

通过两轮 Eval → 分析失败 → 修复 Prompt 的迭代，语义正确率从 **87.5% → 94.12%**：

**第一轮发现的问题及修复：**

| 根因 | 影响题目 | 修复方式 |
| ---- | -------- | -------- |
| LLM 使用 `DATE_SUB()` 等 MySQL 语法 | 多题执行失败 | Prompt 规则：禁止 `DATE_SUB/DATE_ADD`，仅用 `INTERVAL` |
| LLM 生成 SQL 注释被 Guard 拦截 | D7/D30 留存率 | Prompt 规则：禁止输出 `--` 或 `/* */` 注释 |
| `order_reviews` 表不含 `order_purchase_ts` | 评分趋势类问题 | Prompt 规则：review 查询用 `review_creation_date` 锚定 |
| 对比类问题输出宽表而非逐行 | 环比/同比问题 | Prompt 规则：强制使用 `UNION ALL` 格式 |
| 使用 `DATE_FORMAT()` 函数 | 新增买家数 | Prompt 规则：改用 `strftime('%Y-%m', col)` |

输出报告结构：

---

## Statsig + LLM Agent 示例

仓库里补了一份面向 Agent 场景的 Statsig 接入示例，适合参考下面几类能力怎么接：

- `feature gate` 控制 Agent 是否灰度放量
- `dynamic config` 控制模型、工具开关、最大步数
- `prompt` 作为运行时控制面
- `event` 记录时延、采纳率、运行结果
- `online eval` 回传线上评分

可直接查看：

- [docs/statsig_llm_agent.md](docs/statsig_llm_agent.md)
- [examples/statsig_agent_example.py](examples/statsig_agent_example.py)

这个示例默认不影响当前应用运行，也没有把 Statsig 依赖强行塞进主流程依赖里；如果你要单独跑它，再安装：

```bash
pip install statsig-python-core statsig-ai openai
```

```json
{
  "summary": { "semantic_pass_rate_pct": 94.12, "compliance_pass_rate_pct": 100.0, "..." : "..." },
  "by_difficulty": { "easy": {}, "medium": {}, "hard": {} },
  "by_category": { "gmv": {}, "delivery": {}, "customer": {}, "..." : {} },
  "failures": [ { "id": 5, "failure_stage": "semantic", "failure_detail": "..." } ]
}
```

---

## 架构说明

### LLM 调用链路

```text
用户问题
    ↓
metrics_store.py 加载指标词典（注入 Prompt，约束 GMV/复购率等口径）
    ↓
llm_sql.py 调用 DeepSeek/OpenAI API → 生成 SQL
    ↓（若 API 不可用）
内置 fallback 规则（覆盖 6 类高频问题）
    ↓
_anchor_relative_date()：将 current_date 替换为数据集内最新日期，避免查询超出数据范围
```

### SQL 安全防护（SQLGuard）

每条 SQL 在执行前必须通过以下检查：

- 只允许 `SELECT` / `WITH` 语句开头
- 禁止 `INSERT / UPDATE / DELETE / DROP / ALTER / CREATE` 等写操作关键字
- 禁止 SQL 注释（`--` / `/* */`）
- 只允许访问白名单内的表/视图
- 若缺少 `LIMIT`，自动追加默认值（可在侧边栏调整）

### 指标语义层（metrics.yml）

`metrics/metrics.yml` 定义了 28 个业务指标的中文名、计算公式和 SQL 提示，在每次 LLM 调用时作为 System Prompt 注入，确保：

- GMV 不计算 `canceled` / `unavailable` 订单
- 客单价分母必须是 `DISTINCT order_id`
- 新老客拆分口径统一
- 复购率等复杂指标按正确逻辑生成 SQL
