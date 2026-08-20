# RepoPilot 评估方案

## 评估目标

评估拆成检索、答案证据、任务路由和工程性能四层，重点验证系统是否选对工程工具、命中对应材料、在证据不足时停止推断，并保留可诊断轨迹。

## 基线用例

| 场景 | 示例问题 | 期望工程工具 | 主要检查 |
| --- | --- | --- | --- |
| 架构说明 | 解释 State Graph 与证据验证链路 | agent_planner | 来源、路径与停止条件 |
| 变更影响 | Cross-Encoder 会影响哪些模块 | impact_analysis | 兼容性、资源、回退 |
| 故障诊断 | Embedding 下载超时如何恢复 | incident_triage | 假设、证据、下一步 |
| PR 审查 | 检查兼容性与测试缺口 | pr_review | 错误处理、幂等、回归 |
| 验证计划 | 验证低证据重试与回滚 | validation_plan | baseline、oracle、故障注入 |
| 域外问题 | 生产结算重复扣款根因 | evidence guard | 有界重试并拒绝编造 |
| Memory/Skill/ReAct | 解释分层记忆和反思链路 | agent_planner + reflection | 来源、Skill、Plan、反思 |

## 指标

- 检索：Source Hit Rate、Top-K 命中、MRR、Rerank 分数。
- 证据：Evidence Gate Accuracy、Citation Coverage、Groundedness、低证据拦截率。
- Agent：Intent Accuracy、Tool Accuracy、平均检索次数、Skill、ReAct Steps、Reflection、Stop Reason、Graph Path。
- Context/Memory：Token Budget、Source Diversity、Evidence Relation、Memory Partition 与召回分数。
- 工程：p50/p95 Latency、API 成功率、索引构建时间、失败恢复覆盖。

运行：

```bash
python -m pytest
python -m scripts.evaluate
```

自动评测结果由 `scripts.evaluate` 写入 `reports/evaluation-results.json`。当前 7 条用例只是一组可复现回归基线，不能解释为生产准确率；真实结果以该 JSON 文件的最新时间戳与模型元数据为准。

## 扩展计划

1. 扩充到 50-100 条人工标注的变更、故障和审查样例。
2. 对比 Dense Memory、Chroma 与 Cross-Encoder 配置的质量、延迟和资源。
3. 增加错误日志、PR diff 和架构决策记录的解析器。
4. 对低证据误放行、跨文档串扰、无效 Query Rewrite 和回退失真做 Bad Case 分析。
5. 将评估结果版本化，作为发布门槛而不是展示数字。
