"""
PlantCAD2 和 EVO2 接口调用层

负责调用下游推理服务的各个接口，处理参数映射和错误处理。
"""

import httpx
from typing import Optional, Dict, Any
from loguru import logger

from app.config import settings
from app.prompts.task_prompts import TASK_NAME_MAP

# 任务ID到接口路径和task参数的映射
TASK_API_MAP = {
    101: {"endpoint": "/api/v1/generate", "task_param": None, "service": "evo2"},
    201: {"endpoint": "/embeddings", "task_param": None, "service": "plantcad2"},
    202: {"endpoint": "/variant-score", "task_param": None, "service": "plantcad2"},
    203: {"endpoint": "/masked-predict", "task_param": None, "service": "plantcad2"},
    204: {"endpoint": "/predict", "task_param": "acr_arabidopsis", "service": "plantcad2"},
    205: {"endpoint": "/predict", "task_param": "acr_nine_species", "service": "plantcad2"},
    206: {"endpoint": "/predict", "task_param": "acr_cell_type", "service": "plantcad2"},
    207: {"endpoint": "/predict", "task_param": "expression_on_off", "service": "plantcad2"},
    208: {"endpoint": "/predict", "task_param": "expression_absolute", "service": "plantcad2"},
    209: {"endpoint": "/predict", "task_param": "translation_on_off", "service": "plantcad2"},
    210: {"endpoint": "/predict", "task_param": "translation_absolute", "service": "plantcad2"},
}


def get_service_base_url(service: str) -> str:
    """获取服务基础URL"""
    if service == "evo2":
        return settings.evo2_base_url
    elif service == "plantcad2":
        return settings.plantcad2_base_url
    else:
        raise ValueError(f"未知服务: {service}")


def build_request_body(task_id: int, params: Dict[str, Any], task_param: Optional[str]) -> Dict[str, Any]:
    """
    根据任务ID构建请求体

    Args:
        task_id: 任务ID
        params: LLM提取的参数
        task_param: LoRA任务的task参数值

    Returns:
        请求体字典
    """
    request_body = {}

    # EVO2 基因序列预测生成
    if task_id == 101:
        request_body["prompt"] = params.get("prompt") or params.get("sequence", "")
        if "numTokens" in params:
            request_body["numTokens"] = params["numTokens"]
        if "temperature" in params:
            request_body["temperature"] = params["temperature"]
        if "topK" in params:
            request_body["topK"] = params["topK"]
        if "topP" in params:
            request_body["topP"] = params["topP"]

    # 嵌入提取
    elif task_id == 201:
        request_body["sequence"] = params.get("sequence")
        if "normalize" in params:
            request_body["normalize"] = params["normalize"]

    # 变异打分
    elif task_id == 202:
        request_body["sequence"] = params.get("sequence")
        request_body["position"] = params.get("position")
        request_body["ref_allele"] = params.get("ref_allele")
        request_body["alt_alleles"] = params.get("alt_alleles")

    # 掩码预测
    elif task_id == 203:
        request_body["sequence"] = params.get("sequence")
        request_body["positions"] = params.get("positions")

    # LoRA功能预测 (204-210)
    else:
        request_body["sequence"] = params.get("sequence")
        request_body["task"] = task_param

    return request_body


async def call_downstream_api(task_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用下游接口

    Args:
        task_id: 任务ID (101, 201-210)
        params: LLM提取的参数

    Returns:
        接口响应结果

    Raises:
        ValueError: 任务ID无效
        httpx.HTTPStatusError: 接口返回错误状态码
        httpx.TimeoutException: 请求超时
    """
    if task_id not in TASK_API_MAP:
        raise ValueError(f"无效的任务ID: {task_id}")

    api_config = TASK_API_MAP[task_id]
    endpoint = api_config["endpoint"]
    task_param = api_config["task_param"]
    service = api_config["service"]

    # 构建请求体
    request_body = build_request_body(task_id, params, task_param)

    # 获取服务URL
    base_url = get_service_base_url(service)
    url = f"{base_url}{endpoint}"

    logger.info(f"调用下游接口 | task_id={task_id} service={service} endpoint={endpoint}")
    logger.debug(f"请求URL: {url}")
    logger.debug(f"请求参数: {request_body}")

    # 带重试的请求
    last_error = None
    for attempt in range(settings.max_retries):
        try:
            async with httpx.AsyncClient(timeout=settings.api_timeout) as client:
                response = await client.post(url, json=request_body)
                response.raise_for_status()

                result = response.json()
                logger.info(f"接口调用成功 | task_id={task_id} attempt={attempt + 1}")
                logger.debug(f"响应结果: {result}")
                return result

        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(f"接口调用超时 | task_id={task_id} attempt={attempt + 1}/{settings.max_retries}")
        except httpx.HTTPStatusError as e:
            last_error = e
            logger.warning(f"接口调用错误 | task_id={task_id} status={e.response.status_code} attempt={attempt + 1}/{settings.max_retries}")
            # 4xx错误不重试
            if 400 <= e.response.status_code < 500:
                raise
        except Exception as e:
            last_error = e
            logger.warning(f"接口调用异常 | task_id={task_id} error={str(e)} attempt={attempt + 1}/{settings.max_retries}")

    # 所有重试都失败
    logger.error(f"接口调用最终失败 | task_id={task_id}")
    raise last_error


def format_result_for_user(task_id: int, result: Dict[str, Any]) -> str:
    """
    将接口返回结果格式化为用户可读的消息

    Args:
        task_id: 任务ID
        result: 接口返回结果

    Returns:
        格式化后的消息
    """
    task_name = TASK_NAME_MAP.get(task_id, "未知任务")

    # EVO2 基因序列预测生成
    if task_id == 101:
        generated = result.get("generated_sequence", "")
        if generated:
            preview = generated[:50] + "..." if len(generated) > 50 else generated
            return f"已生成后续序列（长度：{len(generated)}bp）：{preview}"
        return "基因序列预测生成完成"

    # 嵌入提取
    elif task_id == 201:
        shape = result.get("shape", [])
        if shape:
            return f"已成功提取嵌入向量，形状为 {shape[0]}×{shape[1]}"
        return "嵌入提取完成"

    # 变异打分
    elif task_id == 202:
        scores = result.get("scores", {})
        if scores:
            score_str = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
            return f"变异打分结果：{score_str}"
        return "变异打分完成"

    # 掩码预测
    elif task_id == 203:
        predictions = result.get("predictions", {})
        if predictions:
            return f"已完成 {len(predictions)} 个位置的碱基概率预测"
        return "掩码预测完成"

    # 二分类任务 (204, 205, 207, 209)
    elif task_id in [204, 205, 207, 209]:
        prediction = result.get("prediction", "")
        probability = result.get("probability", 0)
        if prediction:
            return f"{task_name}结果：{prediction}（概率：{probability:.1%}）"
        return f"{task_name}完成"

    # 多标签分类任务 (206)
    elif task_id == 206:
        num_labels = result.get("num_labels", 0)
        return f"细胞类型ACR预测完成，共 {num_labels} 个细胞类型的概率"

    # 回归任务 (208, 210)
    elif task_id in [208, 210]:
        prediction = result.get("prediction", 0)
        if prediction is not None:
            return f"{task_name}结果：{prediction:.4f}"
        return f"{task_name}完成"

    return f"{task_name}已完成"
