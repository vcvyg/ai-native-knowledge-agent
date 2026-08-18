# Engineering Change Intelligence Agent

一个面向软件仓库的工程变更分析与故障诊断 Agent。它把架构文档、PR 说明、日志、Runbook 和评测结果组织成可检索证据，并围绕变更影响、PR 风险、故障假设、验证计划和回滚步骤给出有边界的工程结论。

系统核心不是普通问答，而是一条可验证、可中断、可恢复的 Agent 控制流：显式 State Graph 管理节点与状态；Verification Loop 在证据不足时改写查询并有界重试；执行图展示工具调用、失败验证、重试、回退和停止原因，并支持断点、继续、编辑后重跑与终止。

## 主要能力

- **工程证据库**：支持 Markdown、TXT、PDF、DOCX；内容哈希保证重复导入幂等，并保留 title、section、source 和 chunk metadata。
- **真实 Embedding**：默认用 `BAAI/bge-small-zh-v1.5` 生成 512 维归一化稠密向量。
- **向量检索**：支持内存余弦索引与 Chroma；两种后端共享同一 Embedding Provider。
- **可插拔 Reranker**：默认结构/关键词启发式重排，可选延迟加载 Cross-Encoder；失败时保留原始错误、进入降级状态并熔断后续模型重试。
- **工程任务路由**：仓库问答、证据摘要、变更影响、故障诊断、PR 审查、验证计划、Runbook、架构说明、对比与评估。
- **证据门控**：检查原问题与命中材料是否有直接领域信号；低证据时不调用外部 LLM。
- **有界恢复**：Query Rewrite、检索预算和全局图转换上限共同防止无界循环。
- **执行轨迹**：Graph Path、Tool Trace、检索次数、实际 Reranker、证据质量和停止原因均可观察。
- **可视化调试**：节点断点、内存 Checkpoint、继续、编辑查询后从 Route 重启、终止与清理。
- **Session Memory**：保存最近对话上下文，支持围绕同一变更继续追问。

## 工作流

```text
Engineering Query
       ↓
Route Intent
       ↓
Retrieve → Rerank
       ↓
Verify Evidence ── low evidence + budget ─→ Rewrite Query
       │                                         │
       │ evidence sufficient / retry limit       └──→ Retrieve
       ↓
Execute Engineering Tools
   ├─ impact_analysis / incident_triage
   ├─ pr_review / validation_plan / runbook
   ├─ evidence_summary / explain_component
   └─ agent_planner / compare / evaluation
       ↓
Synthesize → Citations + Metrics + Agent Trace

Debug: breakpoint → checkpoint → resume / edit and restart / cancel
```

`AgentState` 保存原始查询、检索查询、意图、候选工具、最佳证据、尝试次数和停止原因。完整设计见 [架构说明](docs/architecture.md)。

## 技术栈

- Backend：Python、FastAPI、Pydantic、Uvicorn
- Retrieval：SentenceTransformers、BGE、Dense Cosine Search、Chroma、Cross-Encoder
- Agent：State Graph、Verification Loop、Checkpoint、Breakpoint、Query Rewrite、Tool Use、Session Memory
- LLM：OpenAI-compatible Chat API Adapter（可选）
- Frontend：HTML、CSS、JavaScript、SVG
- Engineering：Pytest、JSONL Evaluation、Docker Compose、GitHub Actions

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8015 --reload
```

打开 `http://127.0.0.1:8015`。首次启动会下载 `BAAI/bge-small-zh-v1.5`，之后从本地缓存加载。模型下载与知识库索引在后台线程执行，因此服务启动后立即可响应：初始化期间 `/api/health` 返回 `status=initializing`，问答与知识库接口返回 503 提示稍后重试；前端会自动轮询直到就绪。知识库重新加载是增量的：新增或修改的文档只重编码变更部分，未变文档复用已有向量。

默认配置：

```bash
export AI_AGENT_EMBEDDING_PROVIDER=sentence_transformers
export AI_AGENT_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
export AI_AGENT_VECTOR_BACKEND=memory
export AI_AGENT_RERANKER=heuristic
```

启用 Chroma：

```bash
python -m pip install -r requirements-chroma.txt
export AI_AGENT_VECTOR_BACKEND=chroma
```

启用 Cross-Encoder：

```bash
export AI_AGENT_RERANKER=cross_encoder
export AI_AGENT_RERANKER_MODEL=BAAI/bge-reranker-base
```

Cross-Encoder 在首次检索时加载；依赖或模型不可用时回退至 heuristic，并在运行指标中暴露实际策略与错误。检索尝试次数默认为 2，可通过 `AI_AGENT_MAX_RETRIEVAL_ATTEMPTS=1..3` 调整。

容器运行：

```bash
docker compose up --build
```

## 可选 LLM

没有 API Key 时，系统使用本地确定性 synthesizer。接入兼容接口时：

```bash
export AI_AGENT_LLM_BASE_URL="https://api.openai.com/v1"
export AI_AGENT_LLM_API_KEY="your_key"
export AI_AGENT_LLM_MODEL="gpt-4o-mini"
```

只有通过证据门控的请求才会调用外部模型。

## API

```http
GET    /api/health
GET    /api/kb/stats
GET    /api/kb/documents
GET    /api/workflow
POST   /api/kb/upload
DELETE /api/kb/documents/{doc_id}
POST   /api/ask
POST   /api/debug/run
POST   /api/debug/resume
DELETE /api/debug/runs/{run_id}
```

`/api/ask` 返回 `answer`、`intent`、`citations`、`trace` 和 `metrics`。调试接口在节点执行前暂停并保留 Checkpoint；完成或取消后清理。

## 测试与评估

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m mypy
python -m ruff check app scripts tests
python -m scripts.evaluate
python -m scripts.export_posttraining
```

JSONL 基线覆盖架构问答、Cross-Encoder 变更影响、Embedding 下载故障、PR 审查、Verification Loop 验证计划和域外低证据问题。评估同时检查 Intent、Source Hit、Tool、Evidence Gate 与延迟，结果写入 [evaluation-results.json](reports/evaluation-results.json)。

后训练导出器会把 Graph Event、Tool Call、引用、Stop Reason 和可解释奖励写成显式轨迹；默认只训练助手回答与工具选择，屏蔽工具结果和检索证据。训练阶段划分见 [posttraining-roadmap.md](docs/posttraining-roadmap.md)。

这是一组小规模、可复现的工程回归基线，不代表生产数据上的泛化效果。

## 中断与延迟交接

长任务因额度或外部资源中断时，先保存可恢复交接，不使用阻塞式 sleep：

```bash
python -m scripts.deferred_handoff save \
  --objective "完成当前工程任务" \
  --reason "quota unavailable" \
  --delay-minutes 60 \
  --completed "已完成的工作" \
  --remaining "待完成的工作" \
  --next-action "恢复后的第一条命令"

python -m scripts.deferred_handoff ready
```

交接文件默认保存在被忽略的 `.task_handoffs/`，包含 Git 工作区快照并自动脱敏常见凭证。脚本只能判断计划恢复时间是否到达，不能自行感知额度恢复或唤醒 Codex。
