from pydantic import BaseModel
from typing import List


class WebshareSettings(BaseModel):
    base_user: str
    password: str
    host: str
    port: int
    first_profile: int
    max_profile: int


class PoolSettings(BaseModel):
    max_profile_switches_per_request: int
    lease_timeout_seconds: float
    base_sleep_seconds: float
    jitter_seconds: float


class YouTubeSettings(BaseModel):
    default_languages: list[str]
    request_timeout_seconds: float


class LLMSettings(BaseModel):
    system_prompt: str

    model_dir: str
    model_pattern: str

    n_ctx: int
    n_threads: int
    n_gpu_layers: int

    default_repeat_penalty: float
    default_stop: List[str]


class AppSettings(BaseModel):
    webshare: WebshareSettings
    pool: PoolSettings
    youtube: YouTubeSettings
    llm: LLMSettings