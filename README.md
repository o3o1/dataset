# Orchestration SFT Dataset

This project contains a small supervised fine-tuning (SFT) dataset that teaches a model how to orchestrate tool calls for engineering task flows.

## Dataset layout
- `data/orchestration_sft.json` – each element captures a user request, a fuzzy tool roster, and the expected JSON plan mapping tasks to agents.

## Sample schema
```json
{
  "input": {
    "user_request": "自然语言需求",
    "available_tools": [
      {"id": "工具标识", "name": "模糊名称", "description": "触发条件"}
    ]
  },
  "output": [
    {"step_name": "任务1，...", "agent_name": "匹配的工具名"}
  ]
}
```

## Coverage highlights
- 多样化的触发词（如“来吧”、“let's go”、“搞事”等）帮助模型判断是否需要调用统筹类专家。
- 确保硬件专家仅在具体设计场景出现，而非总体方案阶段。
- 包含仿真、文档撰写、咨询、兜底等多角色，强化模型在动态工具列表中的匹配能力。

Run `python main.py` to verify the environment is set up.
