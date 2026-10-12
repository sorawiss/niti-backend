import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


class OpenRouterSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    model: str = "openai/gpt-5-mini"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=1200, ge=1)
    timeout_seconds: float = Field(default=60, gt=0)
    zero_data_retention: bool = True
    allow_fallbacks: bool = False
    app_url: str | None = None
    app_name: str | None = "NITI Legal Chat"


@lru_cache(maxsize=1)
def get_openrouter_settings() -> OpenRouterSettings:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    return OpenRouterSettings(
        api_key=SecretStr(api_key),
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1200")),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        zero_data_retention=_read_bool("LLM_ZERO_DATA_RETENTION", True),
        allow_fallbacks=_read_bool("LLM_ALLOW_FALLBACKS", False),
        app_url=os.getenv("OPENROUTER_APP_URL") or None,
        app_name=os.getenv("OPENROUTER_APP_NAME", "NITI Legal Chat") or None,
    )