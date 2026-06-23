"""
意图识别路由

处理意图识别请求，根据置信度执行不同的处理逻辑。
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.schemas.requests import IntentRequest, IntentResponse, SuggestedTask, AvailableTask, ErrorInfo
from app.services.intent_recognizer import recognize_intent
from app.services.param_extractor import extract_params
from app.services.api_caller import call_downstream_api, format_result_for_user, TASK_API_MAP
from app.prompts.task_prompts import TASK_NAME_MAP, TASK_MODEL_MAP, TASK_DETAILS

router = APIRouter(prefix="/intent", tags=["意图识别"])


@router.post("/recognize", response_model=IntentResponse)
async def recognize_user_intent(request: IntentRequest):
    """
    意图识别接口

    接收用户输入，通过大模型进行意图识别，返回三种置信度场景的结果：

    - **高置信度 (high)**: 直接提取参数并调用下游接口，返回计算结果
    - **中置信度 (medium)**: 返回推荐任务列表和引导消息
    - **低置信度 (low)**: 返回兜底引导和所有可用任务列表
    """
    logger.info(f"收到意图识别请求 | session_id={request.session_id}")
    logger.info(f"用户输入: {request.user_input[:200]}...")

    try:
        # 1. 调用LLM进行意图识别
        intent_result = await recognize_intent(request.user_input)
        confidence = intent_result.get("confidence", "low")

        # 2. 根据置信度处理
        if confidence == "high":
            return await handle_high_confidence(intent_result)
        elif confidence == "medium":
            return handle_medium_confidence(intent_result)
        else:
            return handle_low_confidence(intent_result)

    except Exception as e:
        logger.error(f"意图识别处理异常: {str(e)}", exc_info=True)
        return IntentResponse(
            confidence="low",
            guide_message="抱歉，处理过程中出现错误，请稍后重试。您可以直接选择需要的任务。",
            available_tasks=[
                AvailableTask(task_id=tid, task_name=details["name"], model=details["model"])
                for tid, details in TASK_DETAILS.items()
            ],
            error=ErrorInfo(code=1001, message="意图识别失败", detail=str(e)),
        )


async def handle_high_confidence(intent_result: dict) -> IntentResponse:
    """
    处理高置信度场景

    Args:
        intent_result: 意图识别结果

    Returns:
        包含计算结果的响应
    """
    task_id = intent_result.get("task_id")
    task_name = intent_result.get("task_name", TASK_NAME_MAP.get(task_id, "未知任务"))
    model = intent_result.get("model", TASK_MODEL_MAP.get(task_id, "未知模型"))
    params = intent_result.get("params", {})

    logger.info(f"高置信度场景 | task_id={task_id} task_name={task_name}")

    # 验证任务ID
    if task_id not in TASK_API_MAP:
        logger.warning(f"无效的任务ID: {task_id}")
        return IntentResponse(
            confidence="low",
            guide_message=f"识别到的任务ID({task_id})无效，请重新选择任务。",
            available_tasks=[
                AvailableTask(task_id=tid, task_name=details["name"], model=details["model"])
                for tid, details in TASK_DETAILS.items()
            ],
            error=ErrorInfo(code=1001, message="任务ID无效", detail=f"task_id={task_id}不在支持范围内"),
        )

    # 检查参数是否完整
    if not params or not (params.get("sequence") or params.get("prompt")):
        logger.info("参数不完整，降级为中置信度推荐")
        # 降级为中置信度
        return IntentResponse(
            confidence="medium",
            suggested_tasks=[
                SuggestedTask(
                    task_id=task_id,
                    task_name=task_name,
                    model=model,
                    description=TASK_DETAILS.get(task_id, {}).get("description", ""),
                    guide_message=TASK_DETAILS.get(task_id, {}).get("guide_message", "请提供DNA序列"),
                )
            ],
            guide_message=f"已识别到您需要进行{task_name}，但缺少必要的数据。{TASK_DETAILS.get(task_id, {}).get('guide_message', '请提供DNA序列')}",
        )

    # 调用下游接口
    try:
        result = await call_downstream_api(task_id, params)
        guide_message = format_result_for_user(task_id, result)

        return IntentResponse(
            confidence="high",
            task_id=task_id,
            task_name=task_name,
            model=model,
            params=params,
            result=result,
            guide_message=f"已为您完成{task_name}。{guide_message}",
        )

    except Exception as e:
        logger.error(f"下游接口调用失败 | task_id={task_id} error={str(e)}")
        # 调用失败，降级为推荐任务
        return IntentResponse(
            confidence="medium",
            suggested_tasks=[
                SuggestedTask(
                    task_id=task_id,
                    task_name=task_name,
                    model=model,
                    description=TASK_DETAILS.get(task_id, {}).get("description", ""),
                    guide_message=TASK_DETAILS.get(task_id, {}).get("guide_message", "请提供DNA序列"),
                )
            ],
            guide_message=f"已识别到您需要进行{task_name}，但接口调用暂时失败。请稍后重试或手动调用接口。",
            error=ErrorInfo(code=1003, message="下游接口调用失败", detail=str(e)),
        )


def handle_medium_confidence(intent_result: dict) -> IntentResponse:
    """
    处理中置信度场景

    Args:
        intent_result: 意图识别结果

    Returns:
        包含推荐任务的响应
    """
    suggested_tasks_raw = intent_result.get("suggested_tasks", [])
    guide_message = intent_result.get("guide_message", "您想进行哪种分析？")

    # 转换为SuggestedTask模型
    suggested_tasks = []
    for task in suggested_tasks_raw:
        if isinstance(task, dict):
            suggested_tasks.append(SuggestedTask(
                task_id=task.get("task_id", 0),
                task_name=task.get("task_name", ""),
                model=task.get("model", ""),
                description=task.get("description", ""),
                guide_message=task.get("guide_message", ""),
            ))

    logger.info(f"中置信度场景 | 推荐任务数={len(suggested_tasks)}")

    return IntentResponse(
        confidence="medium",
        suggested_tasks=suggested_tasks,
        guide_message=guide_message,
    )


def handle_low_confidence(intent_result: dict) -> IntentResponse:
    """
    处理低置信度场景

    Args:
        intent_result: 意图识别结果

    Returns:
        包含所有可用任务的响应
    """
    guide_message = intent_result.get("guide_message", "")
    available_tasks_raw = intent_result.get("available_tasks", [])

    # 转换为AvailableTask模型
    available_tasks = []
    for task in available_tasks_raw:
        if isinstance(task, dict):
            available_tasks.append(AvailableTask(
                task_id=task.get("task_id", 0),
                task_name=task.get("task_name", ""),
                model=task.get("model", ""),
            ))

    logger.info(f"低置信度场景 | 可用任务数={len(available_tasks)}")

    return IntentResponse(
        confidence="low",
        guide_message=guide_message,
        available_tasks=available_tasks,
    )
