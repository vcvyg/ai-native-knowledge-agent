# LangGraph Integration

RepoPilot now provides a LangGraph runtime adapter on top of the existing
engineering diagnosis workflow.

## Design

The workflow is represented as an explicit state graph:

```
Planner
  -> Retrieve
  -> Verify Evidence
       | sufficient
       v
     Tool Executor
       -> Reflection
            | complete
            v
         Synthesize
```

## Why an adapter instead of a rewrite

The original workflow already contains bounded transitions, checkpoints,
verification and tool safety rules. LangGraph is introduced as the orchestration
layer while preserving the existing engineering controls.

## Components

- Planner: generates diagnosis steps.
- Retrieve: gathers repository evidence.
- Verify: checks evidence quality before actions.
- Tool Executor: performs repository analysis tools.
- Reflection: decides whether more investigation is required.
- Synthesize: produces the final engineering conclusion.

This exposes RepoPilot as a standard LangGraph Agent workflow while keeping its
existing retrieval, memory and safety modules reusable.
