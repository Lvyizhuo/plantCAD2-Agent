from dataclasses import dataclass, field
import os

@dataclass
class Settings:
    # LLM 配置 (qwen3.7-max via 阿里云百炼)
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen-plus-latest"))

    # 下游服务地址
    plantcad2_base_url: str = field(default_factory=lambda: os.getenv("PLANTCAD2_BASE_URL", "http://localhost:8005"))
    evo2_base_url: str = field(default_factory=lambda: os.getenv("EVO2_BASE_URL", "http://36.137.205.153:8666"))

    # 服务配置
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8010")))

    # 超时配置
    llm_timeout: float = 30.0
    api_timeout: float = 60.0
    max_retries: int = 3

settings = Settings()
