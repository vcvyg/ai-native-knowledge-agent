# 变更请求：为检索服务启用可插拔 Cross-Encoder

## 背景

当前启发式 Reranker 融合向量相似度、关键词覆盖和标题命中，资源开销小但对语义接近的候选区分有限。计划增加 `AI_AGENT_RERANKER=cross_encoder`，使用 `BAAI/bge-reranker-base` 对初召回候选二次排序。

## 约束

默认值继续是 heuristic。Cross-Encoder 只能在首次检索时延迟加载，不能阻塞不需要检索的健康接口。模型或依赖失败时回退至 heuristic，响应指标必须同时暴露 requested strategy、actual strategy 和 fallback error。不得吞掉模型加载原始错误。

## 接口与状态

SearchHit 对外字段保持不变，最终 `score` 仍用于排序和展示，但调用方不得假设新旧策略分数处于同一分布。文档 Embedding 和向量索引不因 Reranker 切换而重建。Session Memory、State Graph 和 Checkpoint schema 不变。

## 验收标准

单元测试覆盖默认策略、非法配置、依赖缺失、模型加载失败和安全回退。固定 JSONL 评测集上 Source Hit 不低于基线，p95 延迟记录但不强行伪造提升。轨迹中可看到 rerank 工具和实际策略；回退后请求仍成功，健康接口暴露 fallback error。

## 发布与回滚

先在离线评测比较质量和延迟，再用小流量启用 Cross-Encoder。监控错误率、p95 延迟、CPU、内存和回退次数。超过预算时把配置恢复为 heuristic 即可回滚，不修改知识库或重建索引。
