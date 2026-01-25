"""Configuration system for Enumeraite."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class Config(BaseModel):
    """Configuration model for Enumeraite application."""
    default_provider: str = Field(default="openai", description="Default AI provider")
    default_count: int = Field(default=50, description="Default number of paths to generate")
    max_concurrent_requests: int = Field(default=20, description="Max concurrent HTTP requests")
    request_timeout: int = Field(default=10, description="HTTP request timeout in seconds")
    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Provider configurations")

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
    # Try to load from specified file
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return Config.from_dict(config_dict)

    # Look for config in standard locations
    standard_paths = [
        "enumeraite.json",
        "~/.enumeraite.json",
        "~/.config/enumeraite/config.json"
    ]

    for path in standard_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            with open(expanded_path, 'r') as f:
                config_dict = json.load(f)
            return Config.from_dict(config_dict)

    # Build config from environment variables
    config_dict = {}

    # OpenAI configuration
    if os.getenv("OPENAI_API_KEY"):
        config_dict.setdefault("providers", {})["openai"] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4")
        }

    # Anthropic/Claude configuration
    if os.getenv("ANTHROPIC_API_KEY"):
        config_dict.setdefault("providers", {})["claude"] = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": os.getenv("CLAUDE_MODEL", "anthropic/claude-sonnet-4")
        }

    # HuggingFace configuration (no API key needed for public models)
    hf_model = os.getenv("HUGGINGFACE_MODEL")
    if hf_model or True:  # Always available
        config_dict.setdefault("providers", {})["huggingface"] = {
            "model": hf_model or "enumeraite/Enumeraite-x-Qwen3-4B-Subdomain",
            "device": os.getenv("HUGGINGFACE_DEVICE", "auto"),
            "max_length": int(os.getenv("HUGGINGFACE_MAX_LENGTH", "512")),
            "temperature": float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.7"))
        }

    return Config.from_dict(config_dict)

def create_default_config(output_path: str = "enumeraite.json") -> None:
    """Create a default configuration file.

    Args:
        output_path: Path where to create the config file
    """
    default_config = {
        "default_provider": "openai",
        "default_count": 50,
        "max_concurrent_requests": 20,
        "request_timeout": 10,
        "providers": {
            "openai": {
                "api_key": "your-openai-api-key-here",
                "model": "gpt-4"
            },
            "claude": {
                "api_key": "your-anthropic-api-key-here",
                "model": "anthropic/claude-sonnet-4"
            },
            "huggingface": {
                "model": "enumeraite/Enumeraite-x-Qwen3-4B-Subdomain",
                "device": "auto",
                "max_length": 512,
                "temperature": 0.7
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(default_config, f, indent=2)

    print(f"Created default configuration file: {output_path}")
    print("Please edit it with your API keys.")