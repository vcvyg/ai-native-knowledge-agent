# RepoPilot｜面向代码仓库的工程变更与故障诊断 Agent

一个面向软件仓库的工程变更分析与故障诊断 Agent。它把架构文档、PR 说明、日志、Runbook 和评测结果组织成可检索证据，并围绕变更影响、PR 风险、故障假设、验证计划和回滚步骤给出有边界的工程结论。

系统核心不是普通问答，而是一条可规划、可执行、可验证、可学习的 Agent 控制流：外层由 **LangGraph StateGraph** 管理流程边界、条件路由与 Checkpoint，内层 Plan → Action → Observation → Reflection 负责动态工具决策；Verification Loop 在证据不足时有界重试，并保留断点、继续、编辑后重跑与终止能力。

## 主要能力

- **Context Engineering**：稳定 document/chunk identity、增量向量更新、软删除/恢复、混合检索、Cross-Encoder、WikiLink Evidence Graph、来源多样性与 token-budget Context Packer。
- **真实 Embedding**：默认用 `BAAI/bge-small-zh-v1.5` 生成 512 维归一化稠密向量。
- **向量检索**：支持内存余弦索引与 Chroma；两种后端共享同一 Embedding Provider。
- **可插拔 Reranker**：默认结构/关键词启发式重排，可选延迟加载 Cross-Encoder；失败时保留原始错误、进入降级状态并熔断后续模型重试。
- **分层 Memory**：Working / Episodic / Semantic / Procedural 四类记忆；支持带稳定来源 ID 的幂等导入，并按 relevance + recency + task type + importance 解释召回。
- **Skill / Tool Registry**：Skill 显式声明 instructions、allowed tools、输入/输出 schema 和 evaluation；统一执行 Native Tool 与 MCP Streamable HTTP Tool。
- **生产安全边界**：写操作进入 Human Approval Gate；Tool Registry 提供 idempotency key、执行状态与补偿 rollback 回调，避免重复副作用。
- **证据门控**：检查原问题与命中材料是否有直接领域信号；低证据时不调用外部 LLM。
- **有界恢复**：Query Rewrite、检索预算和全局图转换上限共同防止无界循环。
- **LangGraph 原生编排**：9 个业务节点直接注册到 LangGraph，条件边负责证据重试与 ReAct 循环，`InMemorySaver` + `interrupt/Command` 负责断点和恢复，不再维护第二套自研图执行器。
- **执行轨迹**：Graph Path、Tool Trace、检索次数、实际 Reranker、证据质量和停止原因均可观察。
- **实时运行事件**：`/api/ask/stream` 通过 SSE 在节点完成时推送 Route、Retrieve、Tool、Reflection 与 Final 事件，而不是等整次请求结束后一次性返回。
- **可视化调试**：节点断点、内存 Checkpoint、继续、编辑查询后从 Route 重启、终止与清理。
- **Agentic Post-training 数据层**：把 Plan、Tool Call、Observation、Reflection、证据和 Stop Reason 导出为 trajectory、SFT tool-policy 样本与同任务 preference pair；规则奖励覆盖来源、工具、证据门控、预算和审批安全。
- **Post-training Dashboard**：在同一 Web UI 展示评测质量、轨迹、奖励分布、SFT/Preference 数据量、训练阶段门禁与生成产物；没有实际训练 run 时明确隐藏 loss、GPU 和 checkpoint 指标。

## 工作流

```text
Engineering Query
       ↓
Route Intent
       ↓
Recall Memory → Plan / Select Skill
       ↓
Hybrid Retrieve → Rerank → Evidence Graph → Context Pack
       ↓
Verify Evidence ── low evidence + budget ─→ Rewrite Query
       │                                         │
       │ evidence sufficient / retry limit       └──→ Retrieve
       ↓
ReAct: Action → Tool Registry → Observation → Reflection
   ├─ impact_analysis / incident_triage
   ├─ pr_review / validation_plan / runbook
   ├─ evidence_summary / explain_component
   └─ agent_planner / compare / evaluation
       ↓
Synthesize → Citations + Metrics + Agent Trace → Episodic Memory

Debug: breakpoint → checkpoint → resume / edit and restart / cancel
```

`AgentState` 保存原始查询、检索查询、意图、候选工具、最佳证据、尝试次数和停止原因。完整设计见 [架构说明](docs/architecture.md)。

## 技术栈

- Backend：Python、FastAPI、Pydantic、Uvicorn
- Retrieval：SentenceTransformers、BGE、Dense Cosine Search、Chroma、Cross-Encoder
- Agent：LangGraph、StateGraph、Checkpoint / Interrupt、ReAct、Planning / Reflection、Verification Loop、Skill、Native/MCP Tool Registry
- Memory：Working、Episodic、Semantic、Procedural、幂等 Memory Import
- Safety：Human Approval、Idempotency、Rollback、Bounded Retry / Stop Reason
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
export AI_AGENT_CONTEXT_TOKEN_BUDGET=1800
export AI_AGENT_MAX_REACT_STEPS=3
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

配置 MCP Streamable HTTP 工具（密钥只通过运行环境传入，不写入仓库）：

```bash
export AI_AGENT_MCP_TOOLS_JSON='[{"name":"repo_status","endpoint":"http://127.0.0.1:3000/mcp","remote_name":"repo_status","intents":["pr_review"],"risk":"read"}]'
```

`risk=write|destructive` 的 MCP/Native Tool 不会直接执行，而是返回 `human_approval_required`；Tool Registry 只有在审批 token 匹配后才执行，并通过 idempotency key 避免重复提交。

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
GET    /api/skills
GET    /api/tools
GET    /api/memory/stats
GET    /api/training/dashboard
POST   /api/memory/import
POST   /api/kb/upload
PUT    /api/kb/documents/{doc_id}
DELETE /api/kb/documents/{doc_id}
GET    /api/kb/deleted-documents
POST   /api/kb/restore
POST   /api/ask
POST   /api/ask/stream
POST   /api/debug/run
POST   /api/debug/resume
DELETE /api/debug/runs/{run_id}
```

`/api/ask` 返回 `answer`、`intent`、`citations`、`trace` 和 `metrics`；`/api/workflow` 会明确返回 `framework=langgraph`。调试接口通过 LangGraph interrupt 在节点执行前暂停并保留 Checkpoint，使用相同 thread 恢复；完成或取消后清理。

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

后训练导出器会生成 `posttraining-trajectories.jsonl`、`posttraining-sft.jsonl` 和 `posttraining-preferences.jsonl`。SFT 只训练回答与下一步工具策略，屏蔽环境观察；Preference 只接受“同一 prompt、不同 reward”的真实多轨迹配对，不伪造负样本。训练阶段划分见 [posttraining-roadmap.md](docs/posttraining-roadmap.md)。

Web UI 顶部的 `Post-training` 页面直接读取这些报告，并区分“数据已准备”和“模型已训练”。当前仓库没有执行 SFT / GRPO / GSPO 模型训练，所以控制台不会展示虚构的 loss、GPU 利用率或 checkpoint。

## 经验迁移边界

- ContextSeek：稳定身份、增量更新、软删除、WikiLink / Evidence Link Graph、Cross-Encoder 与 Context Packing 已进入 Context Engine。
- Hebb Mind：Memory Partition、稳定来源、metadata 与幂等导入已进入长期记忆层。
- AReno：Agent trajectory、可解释 reward、SFT tool policy 与 preference 数据出口已落地；完整 GSPO/GRPO 训练仍需 Linux + CUDA 和更大规模多轨迹数据，当前不宣称已经完成模型训练。
- 企业医疗项目：审批、幂等、状态边界和 rollback 被迁移为 Tool Execution Safety；当前内置工程工具均为只读，不为了展示审批而制造伪写操作。
- 智慧养老项目：实时消息链路经验迁移为 Agent Run Event Push；本项目采用更适合单向运行轨迹的 SSE，而不是为了复用关键词照搬 WebSocket/STOMP，Redis 持久化事件仍属于后续生产化范围。

这是一组小规模、可复现的工程回归基线，不代表生产数据上的泛化效果。
