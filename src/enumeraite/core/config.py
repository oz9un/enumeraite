"""Configuration system for Enumeraite."""

import json
import math
import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from .bounded_io import read_bounded_regular_file

_MAX_CONFIG_BYTES = 1_000_000


class Config(BaseModel):
    """Configuration model for Enumeraite application."""

    default_provider: str = Field(default="openai", description="Default AI provider")
    default_count: int = Field(
        default=50, ge=1, le=5_000, description="Default number of paths to generate"
    )
    max_concurrent_requests: int = Field(
        default=20, ge=1, le=1_000, description="Max concurrent HTTP requests"
    )
    request_timeout: float = Field(
        default=10, gt=0, le=3_600, description="HTTP request timeout in seconds"
    )
    http_verify_ssl: bool = Field(
        default=True, description="Verify TLS certificates during HTTP validation"
    )
    http_soft_404_threshold: float = Field(
        default=0.92, ge=0, le=1, description="Soft-404 similarity threshold"
    )
    http_baseline_samples: int = Field(
        default=2, ge=1, le=5, description="Miss probes per virtual-router prefix"
    )
    http_baseline_depth: int = Field(
        default=1, ge=0, le=10, description="Virtual-router calibration depth"
    )
    http_max_baseline_groups: int = Field(
        default=32, ge=1, le=100, description="Maximum calibrated path-prefix groups"
    )
    providers: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Provider configurations"
    )

    @field_validator(
        "default_count",
        "max_concurrent_requests",
        "http_baseline_samples",
        "http_baseline_depth",
        "http_max_baseline_groups",
        mode="before",
    )
    @classmethod
    def require_integer_fields(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("resource counts must be JSON integers")
        return value

    @field_validator("request_timeout", "http_soft_404_threshold", mode="before")
    @classmethod
    def require_numeric_fields(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("numeric resource settings must be JSON numbers")
        return value

    @field_validator("http_verify_ssl", mode="before")
    @classmethod
    def require_boolean_fields(cls, value: Any) -> Any:
        if not isinstance(value, bool):
            raise ValueError("http_verify_ssl must be a JSON boolean")
        return value

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config instance from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(**config_dict)

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider configuration dictionary (empty if not found)
        """
        return self.providers.get(provider_name, {})


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment variables.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Config instance with loaded configuration

    The function looks for configuration in this order:
    1. Specified config_path (if provided and exists)
    2. Standard paths: enumeraite.json, ~/.enumeraite.json, ~/.config/enumeraite/config.json
    3. Environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    4. Default configuration
    """
    # Environment variables augment the file, but never replace explicitly
    # configured provider fields.
    config_dict = _load_config_data(config_path)

    providers = config_dict.setdefault("providers", {})
    if not isinstance(providers, dict) or not all(
        isinstance(name, str) and isinstance(value, dict) for name, value in providers.items()
    ):
        raise ValueError("providers must be an object of provider configuration objects")
    confidence = os.getenv("ENUMERAITE_CONFIDENCE", "model")
    context = _env_bool("ENUMERAITE_CONTEXT", True)
    hybrid = _env_bool("ENUMERAITE_HYBRID", True)
    generation_overrides = {
        key: value
        for key, value in {
            "temperature": _env_float("ENUMERAITE_TEMPERATURE", allow_zero=True),
            "batch_size": _env_int("ENUMERAITE_BATCH_SIZE"),
            "max_batch_calls": _env_int("ENUMERAITE_MAX_BATCH_CALLS"),
            "max_tokens": _env_int("ENUMERAITE_MAX_TOKENS"),
            "seed_limit": _env_int("ENUMERAITE_SEED_LIMIT"),
            "seed_char_limit": _env_int("ENUMERAITE_SEED_CHAR_LIMIT"),
            "timeout": _env_float("ENUMERAITE_PROVIDER_TIMEOUT"),
            "hybrid_model_fraction": _env_float(
                "ENUMERAITE_HYBRID_MODEL_FRACTION", allow_zero=True
            ),
        }.items()
        if value is not None
    }
    top_level_overrides = {
        "http_verify_ssl": _env_optional_bool("ENUMERAITE_HTTP_VERIFY_SSL"),
        "http_soft_404_threshold": _env_float(
            "ENUMERAITE_HTTP_SOFT_404_THRESHOLD", allow_zero=True
        ),
        "http_baseline_samples": _env_int("ENUMERAITE_HTTP_BASELINE_SAMPLES"),
        "http_baseline_depth": _env_int("ENUMERAITE_HTTP_BASELINE_DEPTH", allow_zero=True),
        "http_max_baseline_groups": _env_int("ENUMERAITE_HTTP_MAX_BASELINE_GROUPS"),
    }
    for key, value in top_level_overrides.items():
        if key not in config_dict and value is not None:
            config_dict[key] = value

    # OpenAI configuration
    if os.getenv("OPENAI_API_KEY") or "openai" in providers:
        _merge_provider(
            providers,
            "openai",
            {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "confidence": confidence,
                "context": context,
                "hybrid": hybrid,
                **generation_overrides,
            },
        )

    # Anthropic/Claude configuration
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key or "anthropic" in providers or "claude" in providers:
        anthropic_config = {
            "api_key": anthropic_key,
            "model": os.getenv("ANTHROPIC_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")),
            "confidence": confidence,
            "context": context,
            "hybrid": hybrid,
            **generation_overrides,
        }
        names = (
            ("anthropic", "claude")
            if anthropic_key
            else tuple(name for name in ("anthropic", "claude") if name in providers)
        )
        for name in names:
            _merge_provider(providers, name, anthropic_config)

    compatible = {
        "openrouter": ("OPENROUTER_API_KEY", "openai/gpt-4o-mini"),
        "groq": ("GROQ_API_KEY", "llama-3.3-70b-versatile"),  # gitleaks:allow
        "deepseek": ("DEEPSEEK_API_KEY", "deepseek-chat"),
        "together": ("TOGETHER_API_KEY", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        "minimax": ("MINIMAX_API_KEY", "MiniMax-M3"),
    }
    for name, (key_env, default_model) in compatible.items():
        if os.getenv(key_env) or name in providers:
            prefix = name.upper()
            _merge_provider(
                providers,
                name,
                {
                    "preset": name,
                    "api_key": os.getenv(key_env),
                    "model": os.getenv(f"{prefix}_MODEL", default_model),
                    "base_url": os.getenv(f"{prefix}_BASE_URL") or None,
                    "confidence": confidence,
                    "context": context,
                    "hybrid": hybrid,
                    **generation_overrides,
                },
            )

    local_defaults = {
        "ollama": ("http://localhost:11434/v1", "qwen3:8b"),
        "vllm": ("http://localhost:8000/v1", "local-model"),
        "llamacpp": ("http://localhost:8080/v1", "local-model"),
        "local": ("http://localhost:11434/v1", "qwen3:8b"),
    }
    for name, (base_url, model) in local_defaults.items():
        prefix = name.upper()
        if (
            os.getenv(f"{prefix}_MODEL")
            or os.getenv(f"{prefix}_BASE_URL")
            or name in providers
            or name in {"ollama", "local"}
        ):
            _merge_provider(
                providers,
                name,
                {
                    "preset": name,
                    "api_key": "sk-local",
                    "model": os.getenv(f"{prefix}_MODEL", model),
                    "base_url": os.getenv(f"{prefix}_BASE_URL", base_url),
                    "confidence": confidence,
                    "context": context,
                    "hybrid": hybrid,
                    **generation_overrides,
                },
            )

    # HuggingFace configuration (no API key needed for public models)
    hf_model = os.getenv("HUGGINGFACE_MODEL")
    _merge_provider(
        providers,
        "huggingface",
        {
            # Leave the model unset unless the user chose one. Commands know
            # the discovery kind and select the matching pinned fine-tune.
            "model": hf_model or None,
            "device": os.getenv("HUGGINGFACE_DEVICE", "auto"),
            "max_length": _env_int("HUGGINGFACE_MAX_LENGTH") or 512,
            "temperature": _env_float("HUGGINGFACE_TEMPERATURE") or 0.7,
            "trust_remote_code": _env_bool("HUGGINGFACE_TRUST_REMOTE_CODE", False),
            "revision": os.getenv("HUGGINGFACE_REVISION") or None,
            "seed_limit": _env_int("ENUMERAITE_SEED_LIMIT") or 500,
            "seed_char_limit": _env_int("ENUMERAITE_SEED_CHAR_LIMIT") or 50_000,
        },
    )

    return Config.from_dict(config_dict)


def load_merklemap_api_key(config_path: Optional[str] = None) -> str:
    """Read the passive credential without validating unrelated AI settings.

    Use the same file selection and file-over-environment precedence as
    ``load_config``. The key stays outside the serializable ``Config`` model.
    An explicit blank file value disables the source, even if an env key exists.
    """
    config_dict = _load_config_data(config_path)
    key = config_dict.get("MERKLEMAP_API_KEY", os.getenv("MERKLEMAP_API_KEY", ""))
    if not isinstance(key, str):
        raise ValueError("MERKLEMAP_API_KEY must be a JSON string")
    return key.strip()


def _merge_provider(
    providers: Dict[str, Dict[str, Any]], name: str, values: Dict[str, Any]
) -> None:
    current = providers.setdefault(name, {})
    for key, value in values.items():
        if key not in current and value is not None:
            current[key] = value


def _load_config_data(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path:
        return _read_config_file(config_path) if os.path.exists(config_path) else {}
    for path in (
        "enumeraite.json",
        "~/.enumeraite.json",
        "~/.config/enumeraite/config.json",
    ):
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return _read_config_file(expanded_path)
    return {}


def _read_config_file(path: str) -> Dict[str, Any]:
    raw = read_bounded_regular_file(
        path,
        _MAX_CONFIG_BYTES,
        not_regular_message="configuration path must be a regular file",
        too_large_message="configuration file exceeds the 1 MB limit",
    )
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a JSON object")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return _parse_bool(name, value)


def _env_optional_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    return None if value is None else _parse_bool(name, value)


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_int(name: str, allow_zero: bool = False) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {requirement}")
    return parsed


def _env_float(name: str, allow_zero: bool = False) -> Optional[float]:
    value = os.getenv(name)
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or (parsed == 0 and not allow_zero):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {requirement}")
    return parsed


def create_default_config(output_path: str = "enumeraite.json") -> None:
    """Create a default configuration file.

    Args:
        output_path: Path where to create the config file
    """
    default_config = {
        "default_provider": "ollama",
        "default_count": 100,
        "max_concurrent_requests": 20,
        "request_timeout": 10,
        "http_verify_ssl": True,
        "http_soft_404_threshold": 0.92,
        "http_baseline_samples": 2,
        "http_baseline_depth": 1,
        "http_max_baseline_groups": 32,
        "providers": {
            "ollama": {
                "preset": "ollama",
                "api_key": "sk-local",
                "base_url": "http://localhost:11434/v1",
                "model": "qwen3:8b",
                "confidence": "model",
                "context": True,
                "hybrid": True,
                "max_tokens": 4000,
                "temperature": 0.7,
            }
        },
    }

    with open(output_path, "w") as f:
        json.dump(default_config, f, indent=2)

    print(f"Created default configuration file: {output_path}")
    print("Start Ollama or edit the file to configure another provider.")
