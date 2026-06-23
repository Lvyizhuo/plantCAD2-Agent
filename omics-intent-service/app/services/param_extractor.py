"""
参数提取器

从用户输入中提取任务所需的参数。
"""

import json
import re
from typing import Dict, Any, Optional
from loguru import logger
from openai import AsyncOpenAI

from app.config import settings
from app.prompts.task_prompts import PARAM_EXTRACTION_PROMPT, TASK_DETAILS

# 初始化 OpenAI 客户端
client = AsyncOpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)


async def extract_params(user_input: str, task_id: int) -> Dict[str, Any]:
    """
    从用户输入中提取任务参数

    Args:
        user_input: 用户输入的文本
        task_id: 任务ID

    Returns:
        提取的参数字典
    """
    logger.info(f"开始参数提取 | task_id={task_id}")

    # 获取任务需要的参数字段
    task_detail = TASK_DETAILS.get(task_id, {})
    required_fields = task_detail.get("data_fields", [])

    # 构建提取提示
    extraction_prompt = f"""{PARAM_EXTRACTION_PROMPT}

当前任务ID: {task_id}
任务名称: {task_detail.get('name', '未知')}
需要提取的参数: {', '.join(required_fields)}

用户输入:
{user_input}

请提取参数并输出JSON格式："""

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "你是一个精确的参数提取助手。只输出JSON格式的参数，不要有任何其他文字。"},
                {"role": "user", "content": extraction_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
            timeout=settings.llm_timeout,
        )

        content = response.choices[0].message.content.strip()
        logger.debug(f"参数提取LLM响应: {content}")

        # 解析JSON
        params = parse_params_response(content)

        # 验证和清理参数
        params = validate_params(params, task_id)

        logger.info(f"参数提取完成 | params_keys={list(params.keys())}")
        return params

    except Exception as e:
        logger.error(f"参数提取异常: {str(e)}")
        # 尝试使用正则表达式提取DNA序列
        return extract_params_by_regex(user_input, task_id)


def parse_params_response(content: str) -> Dict[str, Any]:
    """
    解析LLM返回的参数JSON

    Args:
        content: LLM返回的文本

    Returns:
        参数字典
    """
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试提取JSON块
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试查找{...}
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(content[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return {}


def validate_params(params: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    """
    验证和清理提取的参数

    Args:
        params: 原始参数
        task_id: 任务ID

    Returns:
        验证后的参数
    """
    validated = {}

    # 验证DNA序列
    sequence = params.get("sequence") or params.get("prompt", "")
    if sequence:
        # 清理序列：只保留IUPAC碱基字符
        sequence = sequence.upper().strip()
        sequence = re.sub(r'[^ACGTNRYSWKMBDHV]', '', sequence)
        if sequence:
            if task_id == 101:
                validated["prompt"] = sequence
            else:
                validated["sequence"] = sequence

    # 验证变异打分参数 (202)
    if task_id == 202:
        if "position" in params:
            try:
                validated["position"] = int(params["position"])
            except (ValueError, TypeError):
                pass
        if "ref_allele" in params:
            ref = str(params["ref_allele"]).upper().strip()
            if ref in ["A", "C", "G", "T"]:
                validated["ref_allele"] = ref
        if "alt_alleles" in params:
            alts = params["alt_alleles"]
            if isinstance(alts, str):
                alts = [alts]
            validated["alt_alleles"] = [a.upper().strip() for a in alts if str(a).upper().strip() in ["A", "C", "G", "T"]]

    # 验证掩码预测参数 (203)
    if task_id == 203:
        if "positions" in params:
            positions = params["positions"]
            if isinstance(positions, list):
                validated["positions"] = [int(p) for p in positions if isinstance(p, (int, float)) or (isinstance(p, str) and p.isdigit())]

    # 验证嵌入提取参数 (201)
    if task_id == 201:
        if "normalize" in params:
            validated["normalize"] = bool(params["normalize"])

    # 验证EVO2参数 (101)
    if task_id == 101:
        if "numTokens" in params:
            try:
                validated["numTokens"] = max(1, min(10000, int(params["numTokens"])))
            except (ValueError, TypeError):
                pass
        if "temperature" in params:
            try:
                validated["temperature"] = max(0, min(2, float(params["temperature"])))
            except (ValueError, TypeError):
                pass
        if "topK" in params:
            try:
                validated["topK"] = max(1, int(params["topK"]))
            except (ValueError, TypeError):
                pass
        if "topP" in params:
            try:
                validated["topP"] = max(0, min(1, float(params["topP"])))
            except (ValueError, TypeError):
                pass

    return validated


def extract_params_by_regex(user_input: str, task_id: int) -> Dict[str, Any]:
    """
    使用正则表达式作为后备方案提取参数

    Args:
        user_input: 用户输入
        task_id: 任务ID

    Returns:
        提取的参数
    """
    logger.info(f"使用正则表达式提取参数 | task_id={task_id}")
    params = {}

    # 提取DNA序列
    dna_pattern = r'[ACGTNRYSWKMBDHVacgtnryswkmbdhv]{10,}'
    dna_match = re.search(dna_pattern, user_input)
    if dna_match:
        sequence = dna_match.group().upper()
        if task_id == 101:
            params["prompt"] = sequence
        else:
            params["sequence"] = sequence

    # 提取位置信息
    if task_id == 202:
        pos_pattern = r'(?:位置|position|pos)[：:\s]*(\d+)'
        pos_match = re.search(pos_pattern, user_input, re.IGNORECASE)
        if pos_match:
            params["position"] = int(pos_match.group(1))

        # 提取碱基信息
        ref_pattern = r'(?:参考|ref)[：:\s]*([ACGT])'
        ref_match = re.search(ref_pattern, user_input, re.IGNORECASE)
        if ref_match:
            params["ref_allele"] = ref_match.group(1).upper()

        alt_pattern = r'(?:变异|alt|突变)[：:\s]*([ACGT](?:[,\s/]+[ACGT])*)'
        alt_match = re.search(alt_pattern, user_input, re.IGNORECASE)
        if alt_match:
            alts = re.findall(r'[ACGT]', alt_match.group(1).upper())
            if alts:
                params["alt_alleles"] = alts

    if task_id == 203:
        # 提取多个位置
        pos_list_pattern = r'(?:位置|positions?)[：:\s]*\[(.+?)\]'
        pos_match = re.search(pos_list_pattern, user_input, re.IGNORECASE)
        if pos_match:
            positions = re.findall(r'\d+', pos_match.group(1))
            params["positions"] = [int(p) for p in positions]

    return params
