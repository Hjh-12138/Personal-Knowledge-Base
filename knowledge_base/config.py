from dataclasses import dataclass, field
from typing import Optional
import os


DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b",
    "vllm": "mistral-7b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


@dataclass
class Config:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    model_backend: str = "ollama"  # ollama | vllm | openai | anthropic
    model_name: str = ""  # 空字符串时根据 model_backend 使用对应默认值
    api_base_url: str = ""  # vLLM endpoint (http://localhost:8000/v1) or custom OpenAI-compatible
    api_key: str = ""  # API key for openai/anthropic; vLLM defaults to "not-needed"
    temperature: float = 0.3
    embedding_model: str = "all-MiniLM-L6-v2"
    timeout: int = 60
    max_agent_iterations: int = 5
    chroma_persist_dir: str = "./chroma_db"
    documents_dir: str = "./documents"
    debug: bool = False

    def __post_init__(self):
        if not self.model_name:
            self.model_name = DEFAULT_MODELS.get(self.model_backend, "qwen2.5:7b")

    @classmethod
    def from_env(cls) -> "Config":
        env_overrides = {}
        for field_name in cls.__dataclass_fields__:
            env_key = f"KB_{field_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                field_type = type(getattr(cls, field_name))
                if field_type == bool:
                    env_overrides[field_name] = env_val.lower() in ("1", "true", "yes")
                elif field_type == int:
                    env_overrides[field_name] = int(env_val)
                elif field_type == float:
                    env_overrides[field_name] = float(env_val)
                else:
                    env_overrides[field_name] = env_val
        return cls(**env_overrides)
