from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# 请求模型
class IntentRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=10000, description="用户输入的问题文本")
    session_id: Optional[str] = Field(None, description="会话ID，用于上下文关联")

# 推荐任务
class SuggestedTask(BaseModel):
    task_id: int
    task_name: str
    model: str
    description: str
    guide_message: str

# 可用任务（低置信度）
class AvailableTask(BaseModel):
    task_id: int
    task_name: str
    model: str

# 错误信息
class ErrorInfo(BaseModel):
    code: int
    message: str
    detail: Optional[str] = None

# 响应模型
class IntentResponse(BaseModel):
    confidence: str = Field(..., pattern="^(high|medium|low)$")
    task_id: Optional[int] = None
    task_name: Optional[str] = None
    model: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    suggested_tasks: Optional[List[SuggestedTask]] = None
    guide_message: str
    available_tasks: Optional[List[AvailableTask]] = None
    error: Optional[ErrorInfo] = None

# 健康检查响应
class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "omics-intent-service"
