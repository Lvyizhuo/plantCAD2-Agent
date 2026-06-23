# 组学智能体意图识别服务

## 概述

组学智能体意图识别服务是"农业大模型"项目的一部分，负责在主智能体跳转到组学智能体后进行二次意图识别，确定用户具体需要调用哪个任务接口。

## 功能特性

- **高置信度场景**: 直接提取参数并调用下游接口，返回计算结果
- **中置信度场景**: 推荐4-5个相关任务，引导用户提供数据
- **低置信度场景**: 展示所有可用任务，兜底引导

## 支持的任务

| 任务ID | 任务名称 | 模型 |
|--------|---------|------|
| 101 | 基因序列预测生成 | EVO2 |
| 201 | 嵌入提取 | PlantCAD2 |
| 202 | 变异打分 | PlantCAD2 |
| 203 | 掩码预测 | PlantCAD2 |
| 204 | ACR预测-拟南芥 | PlantCAD2 |
| 205 | ACR预测-九物种 | PlantCAD2 |
| 206 | ACR预测-细胞类型 | PlantCAD2 |
| 207 | 表达量预测-开/关 | PlantCAD2 |
| 208 | 表达量预测-绝对值 | PlantCAD2 |
| 209 | 翻译效率预测-开/关 | PlantCAD2 |
| 210 | 翻译效率预测-绝对值 | PlantCAD2 |

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| LLM_API_KEY | 大模型API Key | (必填) |
| LLM_BASE_URL | 大模型服务地址 | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| LLM_MODEL | 大模型名称 | qwen-plus-latest |
| PLANTCAD2_BASE_URL | PlantCAD2服务地址 | http://localhost:8005 |
| EVO2_BASE_URL | EVO2服务地址 | http://36.137.205.153:8666 |

## 启动服务

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export LLM_API_KEY="your-api-key"

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8010
```

## API接口

### POST /intent/recognize

意图识别接口。

**请求参数:**
```json
{
  "user_input": "帮我预测这段DNA序列的表达量开/关：ACGT...",
  "session_id": "session_123"
}
```

**响应示例 (高置信度):**
```json
{
  "confidence": "high",
  "task_id": 207,
  "task_name": "表达量预测-开/关",
  "model": "PlantCAD2",
  "params": {"sequence": "ACGT...", "task": "expression_on_off"},
  "result": {"task": "expression_on_off", "prediction": "POSITIVE", "probability": 0.87},
  "guide_message": "已为您完成表达量预测-开/关。表达量预测-开/关结果：POSITIVE（概率：87.0%）"
}
```

### GET /health

健康检查接口。
