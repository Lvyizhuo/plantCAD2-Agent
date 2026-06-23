"""
意图识别引擎

通过调用大模型进行意图识别，返回置信度和识别结果。
"""

import json
from typing import Dict, Any, Optional
from loguru import logger
from openai import AsyncOpenAI

from app.config import settings
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.prompts.task_prompts import TASK_DETAILS, TASK_NAME_MAP, TASK_MODEL_MAP

# 初始化 OpenAI 客户端（兼容阿里云百炼）
client = AsyncOpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)


async def recognize_intent(user_input: str) -> Dict[str, Any]:
    """
    调用LLM进行意图识别

    Args:
        user_input: 用户输入的文本

    Returns:
        意图识别结果，包含 confidence, task_id, task_name, model, params 等
    """
    logger.info(f"开始意图识别 | user_input={user_input[:100]}...")

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,  # 低温度以获得更确定的结果
            max_tokens=1000,
            timeout=settings.llm_timeout,
        )

        content = response.choices[0].message.content.strip()
        logger.debug(f"LLM原始响应: {content}")

        # 解析JSON响应
        result = parse_llm_response(content)

        # 补充任务信息
        result = enrich_intent_result(result)

        logger.info(f"意图识别完成 | confidence={result.get('confidence')} task_id={result.get('task_id')}")
        return result

    except Exception as e:
        logger.error(f"意图识别异常: {str(e)}")
        return {
            "confidence": "low",
            "guide_message": "抱歉，意图识别服务暂时出现问题，请稍后重试。您可以直接选择需要的任务。",
            "error": str(e),
        }


def parse_llm_response(content: str) -> Dict[str, Any]:
    """
    解析LLM返回的JSON响应

    Args:
        content: LLM返回的文本内容

    Returns:
        解析后的字典
    """
    # 尝试直接解析JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试提取JSON块（可能被包裹在```json...```中）
    import re
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试查找第一个{到最后一个}之间的内容
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(content[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    # 解析失败，返回低置信度
    logger.warning(f"LLM响应解析失败: {content[:200]}")
    return {
        "confidence": "low",
        "guide_message": "抱歉，我无法理解您的意图。请选择需要的任务并提供相应数据。",
    }


def enrich_intent_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    补充意图识别结果的任务信息

    Args:
        result: LLM返回的原始结果

    Returns:
        补充后的结果
    """
    confidence = result.get("confidence", "low")

    # 高置信度：补充任务名称和模型
    if confidence == "high":
        task_id = result.get("task_id")
        if task_id and task_id in TASK_DETAILS:
            result["task_name"] = TASK_NAME_MAP.get(task_id, "未知任务")
            result["model"] = TASK_MODEL_MAP.get(task_id, "未知模型")

            # 如果没有guide_message，生成一个
            if "guide_message" not in result:
                result["guide_message"] = f"正在为您执行{TASK_NAME_MAP[task_id]}任务..."

    # 中置信度：补充推荐任务的详细信息
    elif confidence == "medium":
        suggested_ids = result.get("suggested_tasks", [])
        if suggested_ids and isinstance(suggested_ids[0], int):
            # 将任务ID列表转换为详细信息列表
            suggested_details = []
            for tid in suggested_ids:
                if tid in TASK_DETAILS:
                    detail = TASK_DETAILS[tid]
                    suggested_details.append({
                        "task_id": tid,
                        "task_name": detail["name"],
                        "model": detail["model"],
                        "description": detail["description"],
                        "guide_message": detail["guide_message"],
                    })
            result["suggested_tasks"] = suggested_details

        # 如果没有guide_message，生成一个
        if "guide_message" not in result:
            result["guide_message"] = "您想进行哪种分析？请提供DNA序列数据。"

    # 低置信度：补充可用任务列表
    else:
        if "available_tasks" not in result:
            result["available_tasks"] = [
                {"task_id": tid, "task_name": details["name"], "model": details["model"]}
                for tid, details in TASK_DETAILS.items()
            ]
        if "guide_message" not in result:
            result["guide_message"] = (
                "您好！我是组学智能体，可以为您提供以下基因序列分析服务：\n\n"
                "【PlantCAD2 模型】\n"
                "1. 嵌入提取 - 提取DNA序列的向量表示\n"
                "2. 变异打分 - 评估变异的致病性\n"
                "3. 掩码预测 - 预测指定位置的碱基概率\n"
                "4. ACR预测 - 预测活跃顺式调控元件\n"
                "5. 表达量预测 - 预测基因表达水平\n"
                "6. 翻译效率预测 - 预测翻译效率\n\n"
                "【EVO2 模型】\n"
                "7. 基因序列预测生成 - 预测并生成后续序列\n\n"
                "请选择您需要的任务，并提供相应的DNA序列数据。"
            )

    return result
