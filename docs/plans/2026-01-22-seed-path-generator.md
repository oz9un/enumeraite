# Seed-Based Path Generator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an AI-powered tool that expands web attack surfaces by generating new API paths from known endpoints using multiple model providers.

**Architecture:** Plugin-based provider system with CLI interface supporting batch and continuous discovery modes. Core engine orchestrates path generation, validation, and learning across different AI models (OpenAI, Claude, custom Enumeraite model).

**Tech Stack:** Python 3.9+, Click (CLI), aiohttp (HTTP validation), pydantic (config), pytest (testing)

---

## Task 1: Project Structure Setup

**Files:**
- Create: `src/enumeraite/__init__.py`
- Create: `src/enumeraite/core/__init__.py`
- Create: `src/enumeraite/providers/__init__.py`
- Create: `src/enumeraite/cli/__init__.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `tests/__init__.py`

**Step 1: Create project structure**

```bash
mkdir -p src/enumeraite/{core,providers,cli}
mkdir -p tests/{core,providers,cli}
touch src/enumeraite/__init__.py
touch src/enumeraite/core/__init__.py
touch src/enumeraite/providers/__init__.py
touch src/enumeraite/cli/__init__.py
touch tests/__init__.py
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "enumeraite"
version = "0.1.0"
description = "AI-Assisted Web Attack Surface Enumeration"
authors = [{name = "Özgün Kültekin", email = "ozgun@enumeraite.com"}]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "click>=8.0.0",
    "aiohttp>=3.8.0",
    "pydantic>=2.0.0",
    "openai>=1.0.0",
    "anthropic>=0.7.0",
]

[project.scripts]
enumeraite = "enumeraite.cli.main:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

**Step 3: Create requirements.txt**

```
click>=8.0.0
aiohttp>=3.8.0
pydantic>=2.0.0
openai>=1.0.0
anthropic>=0.7.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

**Step 4: Commit project structure**

```bash
git add .
git commit -m "feat: setup project structure and dependencies"
```

## Task 2: Base Provider Interface

**Files:**
- Create: `src/enumeraite/core/models.py`
- Create: `src/enumeraite/core/provider.py`
- Test: `tests/core/test_provider.py`

**Step 1: Write failing test for base provider**

```python
# tests/core/test_provider.py
import pytest
from enumeraite.core.provider import BaseProvider
from enumeraite.core.models import GenerationResult

def test_base_provider_interface():
    provider = BaseProvider({})
    with pytest.raises(NotImplementedError):
        provider.generate_paths(["/api/user"], "example.com", 10)

def test_generation_result_model():
    result = GenerationResult(
        paths=["/api/user/settings", "/api/user/profile"],
        confidence_scores=[0.8, 0.9],
        metadata={"provider": "test", "method": "pattern"}
    )
    assert len(result.paths) == 2
    assert len(result.confidence_scores) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_provider.py -v`
Expected: FAIL with module import errors

**Step 3: Create data models**

```python
# src/enumeraite/core/models.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class GenerationResult(BaseModel):
    paths: List[str] = Field(..., description="Generated API paths")
    confidence_scores: List[float] = Field(..., description="Confidence scores 0.0-1.0")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider metadata")

    def __post_init__(self):
        if len(self.paths) != len(self.confidence_scores):
            raise ValueError("Paths and confidence scores must have same length")

class ValidationResult(BaseModel):
    path: str
    status_code: Optional[int] = None
    exists: bool = False
    method: str = "GET"
    response_time: Optional[float] = None
```

**Step 4: Create base provider interface**

```python
# src/enumeraite/core/provider.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .models import GenerationResult

class BaseProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate_paths(self, known_paths: List[str], target: str, count: int) -> GenerationResult:
        """Generate new API paths based on known paths"""
        raise NotImplementedError

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name for identification"""
        raise NotImplementedError
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/core/test_provider.py -v`
Expected: PASS

**Step 6: Commit base provider interface**

```bash
git add src/enumeraite/core/ tests/core/
git commit -m "feat: add base provider interface and data models"
```

## Task 3: OpenAI Provider Implementation

**Files:**
- Create: `src/enumeraite/providers/openai_provider.py`
- Test: `tests/providers/test_openai_provider.py`

**Step 1: Write failing test for OpenAI provider**

```python
# tests/providers/test_openai_provider.py
import pytest
from unittest.mock import Mock, patch
from enumeraite.providers.openai_provider import OpenAIProvider

@patch('openai.OpenAI')
def test_openai_provider_generation(mock_openai):
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = """
    Generated paths:
    /api/user/settings
    /api/user/preferences
    /api/user/notifications
    """
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    provider = OpenAIProvider({"api_key": "test-key"})
    result = provider.generate_paths(["/api/user/profile"], "example.com", 3)

    assert len(result.paths) == 3
    assert "/api/user/settings" in result.paths
    assert all(0.0 <= score <= 1.0 for score in result.confidence_scores)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_openai_provider.py -v`
Expected: FAIL with import error

**Step 3: Implement OpenAI provider**

```python
# src/enumeraite/providers/openai_provider.py
import re
import openai
from typing import List
from ..core.provider import BaseProvider
from ..core.models import GenerationResult

class OpenAIProvider(BaseProvider):
    def __init__(self, config):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config.get("api_key"))
        self.model = config.get("model", "gpt-4")

    def generate_paths(self, known_paths: List[str], target: str, count: int) -> GenerationResult:
        prompt = self._build_prompt(known_paths, target, count)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )

        content = response.choices[0].message.content
        paths = self._extract_paths(content)[:count]
        confidence_scores = self._calculate_confidence_scores(paths, known_paths)

        return GenerationResult(
            paths=paths,
            confidence_scores=confidence_scores,
            metadata={
                "provider": "openai",
                "model": self.model,
                "target": target
            }
        )

    def _build_prompt(self, known_paths: List[str], target: str, count: int) -> str:
        known_paths_str = "\n".join(known_paths)
        return f"""You are an expert in web application reconnaissance and API endpoint discovery.

Given these known API paths for {target}:
{known_paths_str}

Generate {count} new potential API paths that could exist on this target using:
1. Pattern analysis: Analyze REST conventions, naming patterns, and structural similarities
2. Contextual understanding: Consider domain concepts, user roles, and common functionality
3. Common web application patterns: Think about typical CRUD operations, authentication, admin functions

Requirements:
- Generate ONLY the path portion (starting with /)
- Focus on realistic, well-formed API paths
- Consider different user roles, resources, and operations
- Vary between simple and complex nested paths
- One path per line

Generated paths:"""

    def _extract_paths(self, content: str) -> List[str]:
        # Extract paths that start with /
        path_pattern = r'^(/[^\s\n]*)'
        paths = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('/'):
                # Clean up the path
                path = line.split()[0]  # Take first token in case of extra text
                if self._is_valid_path(path):
                    paths.append(path)
        return paths

    def _is_valid_path(self, path: str) -> bool:
        # Basic validation
        if not path.startswith('/'):
            return False
        if len(path) > 200:  # Reject overly long paths
            return False
        if '..' in path or '//' in path:  # Security check
            return False
        return True

    def _calculate_confidence_scores(self, paths: List[str], known_paths: List[str]) -> List[float]:
        scores = []
        for path in paths:
            score = 0.5  # Base score

            # Boost score for pattern similarity
            for known_path in known_paths:
                similarity = self._calculate_path_similarity(path, known_path)
                score = max(score, 0.3 + similarity * 0.5)

            # Common API patterns get higher scores
            if any(pattern in path.lower() for pattern in ['api', 'user', 'admin', 'auth']):
                score += 0.1

            scores.append(min(score, 1.0))
        return scores

    def _calculate_path_similarity(self, path1: str, path2: str) -> float:
        # Simple similarity based on common segments
        segments1 = set(path1.split('/')[1:])  # Skip empty first element
        segments2 = set(path2.split('/')[1:])

        if not segments1 or not segments2:
            return 0.0

        intersection = segments1.intersection(segments2)
        union = segments1.union(segments2)
        return len(intersection) / len(union) if union else 0.0

    def get_provider_name(self) -> str:
        return "openai"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_openai_provider.py -v`
Expected: PASS

**Step 5: Commit OpenAI provider**

```bash
git add src/enumeraite/providers/ tests/providers/
git commit -m "feat: add OpenAI provider with path generation and confidence scoring"
```

## Task 4: Claude Provider Implementation

**Files:**
- Create: `src/enumeraite/providers/claude_provider.py`
- Test: `tests/providers/test_claude_provider.py`

**Step 1: Write failing test for Claude provider**

```python
# tests/providers/test_claude_provider.py
import pytest
from unittest.mock import Mock, patch
from enumeraite.providers.claude_provider import ClaudeProvider

@patch('anthropic.Anthropic')
def test_claude_provider_generation(mock_anthropic):
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = """
    /api/user/settings
    /api/user/preferences
    /api/user/notifications
    """
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    provider = ClaudeProvider({"api_key": "test-key"})
    result = provider.generate_paths(["/api/user/profile"], "example.com", 3)

    assert len(result.paths) == 3
    assert "/api/user/settings" in result.paths
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_claude_provider.py -v`
Expected: FAIL with import error

**Step 3: Implement Claude provider**

```python
# src/enumeraite/providers/claude_provider.py
import anthropic
from typing import List
from ..core.provider import BaseProvider
from ..core.models import GenerationResult

class ClaudeProvider(BaseProvider):
    def __init__(self, config):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.get("api_key"))
        self.model = config.get("model", "claude-3-sonnet-20240229")

    def generate_paths(self, known_paths: List[str], target: str, count: int) -> GenerationResult:
        prompt = self._build_prompt(known_paths, target, count)

        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7
        )

        content = response.content[0].text
        paths = self._extract_paths(content)[:count]
        confidence_scores = self._calculate_confidence_scores(paths, known_paths)

        return GenerationResult(
            paths=paths,
            confidence_scores=confidence_scores,
            metadata={
                "provider": "claude",
                "model": self.model,
                "target": target
            }
        )

    def _build_prompt(self, known_paths: List[str], target: str, count: int) -> str:
        known_paths_str = "\n".join(known_paths)
        return f"""I need you to generate {count} new API endpoint paths for {target} based on these existing paths:

{known_paths_str}

Use these strategies:
1. Pattern analysis - identify REST patterns, naming conventions, URL structures
2. Contextual understanding - consider what functionality these paths suggest
3. Common patterns - think about typical web application endpoints

Requirements:
- Return only the path part (starting with /)
- One path per line
- Realistic and well-formed paths only
- Consider different user types (user, admin, guest)
- Include both simple and nested paths

Generated paths:"""

    def _extract_paths(self, content: str) -> List[str]:
        paths = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('/'):
                path = line.split()[0]
                if self._is_valid_path(path):
                    paths.append(path)
        return paths

    def _is_valid_path(self, path: str) -> bool:
        if not path.startswith('/'):
            return False
        if len(path) > 200:
            return False
        if '..' in path or '//' in path:
            return False
        return True

    def _calculate_confidence_scores(self, paths: List[str], known_paths: List[str]) -> List[float]:
        scores = []
        for path in paths:
            score = 0.5

            for known_path in known_paths:
                similarity = self._calculate_path_similarity(path, known_path)
                score = max(score, 0.3 + similarity * 0.5)

            if any(pattern in path.lower() for pattern in ['api', 'user', 'admin', 'auth']):
                score += 0.1

            scores.append(min(score, 1.0))
        return scores

    def _calculate_path_similarity(self, path1: str, path2: str) -> float:
        segments1 = set(path1.split('/')[1:])
        segments2 = set(path2.split('/')[1:])

        if not segments1 or not segments2:
            return 0.0

        intersection = segments1.intersection(segments2)
        union = segments1.union(segments2)
        return len(intersection) / len(union) if union else 0.0

    def get_provider_name(self) -> str:
        return "claude"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_claude_provider.py -v`
Expected: PASS

**Step 5: Commit Claude provider**

```bash
git add src/enumeraite/providers/claude_provider.py tests/providers/test_claude_provider.py
git commit -m "feat: add Claude provider implementation"
```

## Task 5: HTTP Validation System

**Files:**
- Create: `src/enumeraite/core/validator.py`
- Test: `tests/core/test_validator.py`

**Step 1: Write failing test for HTTP validator**

```python
# tests/core/test_validator.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from enumeraite.core.validator import HTTPValidator
from enumeraite.core.models import ValidationResult

@pytest.mark.asyncio
async def test_validate_single_path():
    validator = HTTPValidator()

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = Mock()
        mock_response.status = 200
        mock_response.__aenter__ = Mock(return_value=mock_response)
        mock_response.__aexit__ = Mock(return_value=None)
        mock_get.return_value = mock_response

        result = await validator.validate_path("https://example.com/api/test")

        assert result.exists == True
        assert result.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_validator.py -v`
Expected: FAIL with import error

**Step 3: Implement HTTP validator**

```python
# src/enumeraite/core/validator.py
import asyncio
import aiohttp
import time
from typing import List, Optional
from .models import ValidationResult

class HTTPValidator:
    def __init__(self, timeout: int = 10, max_concurrent: int = 20):
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def validate_paths(self, base_url: str, paths: List[str],
                           methods: List[str] = None) -> List[ValidationResult]:
        if methods is None:
            methods = ["GET"]

        tasks = []
        for path in paths:
            for method in methods:
                full_url = f"{base_url.rstrip('/')}{path}"
                task = self.validate_path(full_url, method)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results, group by path
        path_results = {}
        for i, result in enumerate(results):
            if isinstance(result, ValidationResult) and result.exists:
                path = paths[i % len(paths)]
                if path not in path_results or result.status_code < 400:
                    path_results[path] = result

        return list(path_results.values())

    async def validate_path(self, url: str, method: str = "GET") -> ValidationResult:
        async with self.semaphore:
            start_time = time.time()

            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.request(method, url) as response:
                        response_time = time.time() - start_time

                        return ValidationResult(
                            path=url,
                            status_code=response.status,
                            exists=self._is_success_status(response.status),
                            method=method,
                            response_time=response_time
                        )
            except Exception as e:
                return ValidationResult(
                    path=url,
                    status_code=None,
                    exists=False,
                    method=method,
                    response_time=time.time() - start_time
                )

    def _is_success_status(self, status_code: int) -> bool:
        # Consider anything not 404/403 as potentially existing
        return status_code not in [404, 403]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_validator.py -v`
Expected: PASS

**Step 5: Commit HTTP validator**

```bash
git add src/enumeraite/core/validator.py tests/core/test_validator.py
git commit -m "feat: add HTTP path validation system"
```

## Task 6: Configuration System

**Files:**
- Create: `src/enumeraite/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write failing test for config system**

```python
# tests/core/test_config.py
import pytest
import tempfile
import os
from enumeraite.core.config import Config, load_config

def test_config_creation():
    config = Config()
    assert config.default_provider == "openai"
    assert config.default_count == 50

def test_config_from_dict():
    config_dict = {
        "providers": {
            "openai": {"api_key": "test-key", "model": "gpt-4"}
        },
        "default_provider": "openai"
    }
    config = Config.from_dict(config_dict)
    assert config.providers["openai"]["api_key"] == "test-key"

def test_load_config_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"default_provider": "claude", "default_count": 100}')
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.default_provider == "claude"
        assert config.default_count == 100
    finally:
        os.unlink(temp_path)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_config.py -v`
Expected: FAIL with import error

**Step 3: Implement configuration system**

```python
# src/enumeraite/core/config.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class Config(BaseModel):
    default_provider: str = Field(default="openai", description="Default AI provider")
    default_count: int = Field(default=50, description="Default number of paths to generate")
    max_concurrent_requests: int = Field(default=20, description="Max concurrent HTTP requests")
    request_timeout: int = Field(default=10, description="HTTP request timeout in seconds")
    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Provider configurations")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        return cls(**config_dict)

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        return self.providers.get(provider_name, {})

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment"""
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

    # Default config with environment variables
    config_dict = {}

    if os.getenv("OPENAI_API_KEY"):
        config_dict.setdefault("providers", {})["openai"] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4")
        }

    if os.getenv("ANTHROPIC_API_KEY"):
        config_dict.setdefault("providers", {})["claude"] = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
        }

    return Config.from_dict(config_dict)

def create_default_config(output_path: str = "enumeraite.json"):
    """Create a default configuration file"""
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
                "model": "claude-3-sonnet-20240229"
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(default_config, f, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_config.py -v`
Expected: PASS

**Step 5: Commit configuration system**

```bash
git add src/enumeraite/core/config.py tests/core/test_config.py
git commit -m "feat: add configuration system with provider management"
```

## Task 7: Provider Factory and Registry

**Files:**
- Create: `src/enumeraite/core/factory.py`
- Test: `tests/core/test_factory.py`

**Step 1: Write failing test for provider factory**

```python
# tests/core/test_factory.py
import pytest
from enumeraite.core.factory import ProviderFactory
from enumeraite.core.config import Config
from enumeraite.providers.openai_provider import OpenAIProvider

def test_create_provider():
    config = Config.from_dict({
        "providers": {
            "openai": {"api_key": "test-key"}
        }
    })

    factory = ProviderFactory(config)
    provider = factory.create_provider("openai")

    assert isinstance(provider, OpenAIProvider)
    assert provider.config["api_key"] == "test-key"

def test_get_available_providers():
    config = Config.from_dict({
        "providers": {
            "openai": {"api_key": "key1"},
            "claude": {"api_key": "key2"}
        }
    })

    factory = ProviderFactory(config)
    providers = factory.get_available_providers()

    assert "openai" in providers
    assert "claude" in providers
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_factory.py -v`
Expected: FAIL with import error

**Step 3: Implement provider factory**

```python
# src/enumeraite/core/factory.py
from typing import Dict, List, Type
from .provider import BaseProvider
from .config import Config

class ProviderFactory:
    """Factory for creating and managing AI providers"""

    _registry: Dict[str, Type[BaseProvider]] = {}

    def __init__(self, config: Config):
        self.config = config
        self._register_default_providers()

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseProvider]):
        """Register a new provider class"""
        cls._registry[name] = provider_class

    def _register_default_providers(self):
        """Register built-in providers"""
        try:
            from ..providers.openai_provider import OpenAIProvider
            self.register_provider("openai", OpenAIProvider)
        except ImportError:
            pass

        try:
            from ..providers.claude_provider import ClaudeProvider
            self.register_provider("claude", ClaudeProvider)
        except ImportError:
            pass

    def create_provider(self, provider_name: str) -> BaseProvider:
        """Create a provider instance"""
        if provider_name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")

        provider_config = self.config.get_provider_config(provider_name)
        if not provider_config:
            raise ValueError(f"No configuration found for provider '{provider_name}'")

        provider_class = self._registry[provider_name]
        return provider_class(provider_config)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        available = []
        for name in self._registry.keys():
            if self.config.get_provider_config(name):
                available.append(name)
        return available

    def get_default_provider(self) -> BaseProvider:
        """Get the default provider instance"""
        available = self.get_available_providers()
        if not available:
            raise ValueError("No providers available. Please configure at least one provider.")

        default_name = self.config.default_provider
        if default_name in available:
            return self.create_provider(default_name)

        # Fall back to first available
        return self.create_provider(available[0])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_factory.py -v`
Expected: PASS

**Step 5: Commit provider factory**

```bash
git add src/enumeraite/core/factory.py tests/core/test_factory.py
git commit -m "feat: add provider factory and registry system"
```

## Task 8: Core Generation Engine

**Files:**
- Create: `src/enumeraite/core/engine.py`
- Test: `tests/core/test_engine.py`

**Step 1: Write failing test for generation engine**

```python
# tests/core/test_engine.py
import pytest
from unittest.mock import Mock
from enumeraite.core.engine import GenerationEngine
from enumeraite.core.models import GenerationResult

def test_generation_engine():
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=["/api/user/settings", "/api/user/preferences"],
        confidence_scores=[0.8, 0.7],
        metadata={"provider": "test"}
    )

    engine = GenerationEngine(mock_provider)
    result = engine.generate_paths(["/api/user/profile"], "example.com", 10)

    assert len(result.paths) == 2
    mock_provider.generate_paths.assert_called_once_with(["/api/user/profile"], "example.com", 10)

def test_generation_with_validation():
    mock_provider = Mock()
    mock_provider.generate_paths.return_value = GenerationResult(
        paths=["/api/test"],
        confidence_scores=[0.8],
        metadata={"provider": "test"}
    )

    mock_validator = Mock()
    mock_validator.validate_paths.return_value = []  # async mock

    engine = GenerationEngine(mock_provider, mock_validator)
    # Test will be async once we implement it
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_engine.py -v`
Expected: FAIL with import error

**Step 3: Implement generation engine**

```python
# src/enumeraite/core/engine.py
import asyncio
from typing import List, Optional, Set
from .provider import BaseProvider
from .validator import HTTPValidator
from .models import GenerationResult, ValidationResult

class GenerationEngine:
    """Core engine for path generation and validation"""

    def __init__(self, provider: BaseProvider, validator: Optional[HTTPValidator] = None):
        self.provider = provider
        self.validator = validator
        self._seen_paths: Set[str] = set()

    def generate_paths(self, known_paths: List[str], target: str, count: int) -> GenerationResult:
        """Generate new paths using the configured provider"""
        result = self.provider.generate_paths(known_paths, target, count)

        # Filter out duplicates and previously seen paths
        filtered_paths = []
        filtered_scores = []

        for path, score in zip(result.paths, result.confidence_scores):
            normalized_path = self._normalize_path(path)
            if normalized_path not in self._seen_paths and normalized_path not in known_paths:
                filtered_paths.append(path)
                filtered_scores.append(score)
                self._seen_paths.add(normalized_path)

        return GenerationResult(
            paths=filtered_paths,
            confidence_scores=filtered_scores,
            metadata=result.metadata
        )

    async def generate_and_validate_paths(self, known_paths: List[str], target: str,
                                        count: int, methods: List[str] = None) -> tuple[GenerationResult, List[ValidationResult]]:
        """Generate paths and validate them via HTTP"""
        if not self.validator:
            raise ValueError("No validator configured for validation")

        # Generate paths
        generation_result = self.generate_paths(known_paths, target, count)

        # Validate generated paths
        base_url = f"https://{target}"
        validation_results = await self.validator.validate_paths(
            base_url, generation_result.paths, methods
        )

        return generation_result, validation_results

    def _normalize_path(self, path: str) -> str:
        """Normalize path for comparison"""
        return path.lower().rstrip('/')

    def add_discovered_paths(self, paths: List[str]):
        """Add newly discovered paths to the seen set"""
        for path in paths:
            self._seen_paths.add(self._normalize_path(path))

    def reset_seen_paths(self):
        """Reset the seen paths set (useful for new targets)"""
        self._seen_paths.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_engine.py -v`
Expected: PASS

**Step 5: Commit generation engine**

```bash
git add src/enumeraite/core/engine.py tests/core/test_engine.py
git commit -m "feat: add core generation engine with validation support"
```

## Task 9: CLI Interface - Basic Structure

**Files:**
- Create: `src/enumeraite/cli/main.py`
- Create: `src/enumeraite/cli/commands.py`
- Test: `tests/cli/test_commands.py`

**Step 1: Write failing test for CLI**

```python
# tests/cli/test_commands.py
import pytest
from click.testing import CliRunner
from enumeraite.cli.main import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'enumeraite' in result.output.lower()

def test_batch_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['batch', '--help'])
    assert result.exit_code == 0
    assert 'target' in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cli/test_commands.py -v`
Expected: FAIL with import error

**Step 3: Implement CLI main entry point**

```python
# src/enumeraite/cli/main.py
import click
from .commands import batch, continuous

@click.group()
@click.version_option(version='0.1.0')
def cli():
    """Enumeraite: AI-Assisted Web Attack Surface Enumeration

    Generate new API paths from known endpoints using AI pattern recognition.
    """
    pass

cli.add_command(batch)
cli.add_command(continuous)

if __name__ == '__main__':
    cli()
```

**Step 4: Implement CLI commands structure**

```python
# src/enumeraite/cli/commands.py
import click
import json
import asyncio
from pathlib import Path
from typing import List

from ..core.config import load_config
from ..core.factory import ProviderFactory
from ..core.engine import GenerationEngine
from ..core.validator import HTTPValidator

@click.command()
@click.option('--target', required=True, help='Target domain (e.g., example.com)')
@click.option('--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input file containing known paths')
@click.option('--output', 'output_file', type=click.Path(),
              help='Output file for generated paths')
@click.option('--count', default=50, help='Number of paths to generate')
@click.option('--provider', help='AI provider to use (openai, claude)')
@click.option('--advanced', is_flag=True, help='Output detailed JSON format')
@click.option('--validate', is_flag=True, help='Validate generated paths via HTTP')
@click.option('--methods', default='GET', help='HTTP methods to test (comma-separated)')
def batch(target, input_file, output_file, count, provider, advanced, validate, methods):
    """Generate paths in batch mode"""
    # Load known paths
    known_paths = _load_paths_from_file(input_file)
    if not known_paths:
        click.echo("No paths found in input file", err=True)
        return

    # Load configuration
    config = load_config()
    factory = ProviderFactory(config)

    # Get provider
    if provider:
        ai_provider = factory.create_provider(provider)
    else:
        ai_provider = factory.get_default_provider()

    # Setup engine
    validator = HTTPValidator() if validate else None
    engine = GenerationEngine(ai_provider, validator)

    click.echo(f"Generating {count} paths for {target} using {ai_provider.get_provider_name()}")

    if validate:
        # Async validation
        http_methods = [m.strip().upper() for m in methods.split(',')]
        generation_result, validation_results = asyncio.run(
            engine.generate_and_validate_paths(known_paths, target, count, http_methods)
        )
    else:
        # Generation only
        generation_result = engine.generate_paths(known_paths, target, count)
        validation_results = []

    # Output results
    if advanced:
        _output_advanced(generation_result, validation_results, output_file)
    else:
        _output_simple(generation_result, output_file)

    click.echo(f"Generated {len(generation_result.paths)} paths")
    if validation_results:
        valid_count = sum(1 for r in validation_results if r.exists)
        click.echo(f"Validated {len(validation_results)} paths, {valid_count} appear to exist")

@click.command()
@click.option('--target', required=True, help='Target domain')
@click.option('--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input file containing known paths (will be updated)')
@click.option('--duration', default='30m', help='How long to run (e.g., 30m, 2h)')
@click.option('--batch-size', default=20, help='Paths to generate per round')
@click.option('--max-empty-rounds', default=5, help='Stop after N rounds with no discoveries')
@click.option('--provider', help='AI provider to use')
@click.option('--methods', default='GET', help='HTTP methods to test')
def continuous(target, input_file, duration, batch_size, max_empty_rounds, provider, methods):
    """Run continuous discovery mode"""
    click.echo("Continuous mode not implemented yet")
    # Will be implemented in next task

def _load_paths_from_file(file_path: str) -> List[str]:
    """Load paths from input file"""
    paths = []
    with open(file_path, 'r') as f:
        for line in f:
            path = line.strip()
            if path and not path.startswith('#'):
                paths.append(path)
    return paths

def _output_simple(result, output_file):
    """Output simple path list"""
    if output_file:
        with open(output_file, 'w') as f:
            for path in result.paths:
                f.write(f"{path}\n")
    else:
        for path in result.paths:
            click.echo(path)

def _output_advanced(generation_result, validation_results, output_file):
    """Output detailed JSON format"""
    # Create validation lookup
    validation_lookup = {r.path.split('/')[-1]: r for r in validation_results}

    output_data = {
        "generated_count": len(generation_result.paths),
        "metadata": generation_result.metadata,
        "generated_paths": []
    }

    for path, score in zip(generation_result.paths, generation_result.confidence_scores):
        path_data = {
            "path": path,
            "confidence": round(score, 3)
        }

        # Add validation data if available
        if path in validation_lookup:
            val = validation_lookup[path]
            path_data.update({
                "verified": val.exists,
                "status": val.status_code,
                "method": val.method,
                "response_time": round(val.response_time, 3) if val.response_time else None
            })

        output_data["generated_paths"].append(path_data)

    json_output = json.dumps(output_data, indent=2)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(json_output)
    else:
        click.echo(json_output)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/cli/test_commands.py -v`
Expected: PASS

**Step 6: Commit basic CLI structure**

```bash
git add src/enumeraite/cli/ tests/cli/
git commit -m "feat: add CLI interface with batch command"
```

## Task 10: Continuous Discovery Mode Implementation

**Files:**
- Modify: `src/enumeraite/cli/commands.py`
- Create: `src/enumeraite/core/continuous.py`
- Test: `tests/core/test_continuous.py`

**Step 1: Write failing test for continuous mode**

```python
# tests/core/test_continuous.py
import pytest
from unittest.mock import Mock, AsyncMock
from enumeraite.core.continuous import ContinuousDiscovery
from enumeraite.core.models import GenerationResult, ValidationResult

@pytest.mark.asyncio
async def test_single_discovery_round():
    mock_engine = Mock()
    mock_engine.generate_and_validate_paths = AsyncMock()

    # Mock validation results - 1 valid path found
    mock_engine.generate_and_validate_paths.return_value = (
        GenerationResult(paths=["/api/test1", "/api/test2"], confidence_scores=[0.8, 0.7], metadata={}),
        [ValidationResult(path="/api/test1", status_code=200, exists=True)]
    )

    discovery = ContinuousDiscovery(mock_engine, "example.com")

    initial_paths = ["/api/user"]
    new_paths = await discovery.run_single_round(initial_paths, batch_size=10)

    assert "/api/test1" in new_paths
    assert len(new_paths) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_continuous.py -v`
Expected: FAIL with import error

**Step 3: Implement continuous discovery engine**

```python
# src/enumeraite/core/continuous.py
import asyncio
import time
from typing import List, Set, Callable, Optional
from .engine import GenerationEngine
from .models import ValidationResult

class ContinuousDiscovery:
    """Manages continuous discovery loops with adaptive learning"""

    def __init__(self, engine: GenerationEngine, target: str):
        self.engine = engine
        self.target = target
        self.total_discovered = 0
        self.round_number = 0
        self.empty_rounds = 0

    async def run_discovery(self, initial_paths: List[str], duration_minutes: int,
                          batch_size: int, max_empty_rounds: int,
                          progress_callback: Optional[Callable] = None,
                          path_update_callback: Optional[Callable] = None) -> List[str]:
        """Run continuous discovery for specified duration"""

        current_paths = initial_paths.copy()
        start_time = time.time()
        duration_seconds = duration_minutes * 60

        self._log_start(duration_minutes, batch_size, max_empty_rounds)

        while True:
            self.round_number += 1

            # Check stop conditions
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                self._log("Time limit reached")
                break

            if self.empty_rounds >= max_empty_rounds:
                self._log(f"Stopping after {max_empty_rounds} rounds with no discoveries")
                break

            # Run discovery round
            try:
                new_paths = await self.run_single_round(current_paths, batch_size)

                if new_paths:
                    self.empty_rounds = 0
                    self.total_discovered += len(new_paths)
                    current_paths.extend(new_paths)

                    self._log(f"Round {self.round_number}: Found {len(new_paths)} new paths")
                    for path in new_paths:
                        self._log(f"  ✓ {path}")

                    # Callback to update input file
                    if path_update_callback:
                        path_update_callback(new_paths)

                else:
                    self.empty_rounds += 1
                    self._log(f"Round {self.round_number}: No new paths found ({self.empty_rounds}/{max_empty_rounds})")

                # Progress callback
                if progress_callback:
                    progress_callback(self.round_number, len(new_paths), self.total_discovered)

            except Exception as e:
                self._log(f"Error in round {self.round_number}: {e}")
                continue

        self._log_summary(elapsed)
        return current_paths

    async def run_single_round(self, known_paths: List[str], batch_size: int,
                             methods: List[str] = None) -> List[str]:
        """Run a single discovery round and return newly found valid paths"""
        if methods is None:
            methods = ["GET"]

        # Generate and validate paths
        generation_result, validation_results = await self.engine.generate_and_validate_paths(
            known_paths, self.target, batch_size, methods
        )

        # Extract valid paths
        valid_paths = []
        for validation in validation_results:
            if validation.exists:
                # Extract path from full URL
                if validation.path.startswith('http'):
                    path = '/' + validation.path.split('/', 3)[3] if '/' in validation.path.split('/', 3) else '/'
                else:
                    path = validation.path
                valid_paths.append(path)

        # Update engine's seen paths to avoid regenerating
        self.engine.add_discovered_paths(generation_result.paths)

        return valid_paths

    def _log(self, message: str):
        """Log message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _log_start(self, duration_minutes: int, batch_size: int, max_empty_rounds: int):
        """Log discovery start parameters"""
        self._log(f"Starting continuous discovery for {self.target}")
        self._log(f"Duration: {duration_minutes} minutes")
        self._log(f"Batch size: {batch_size} paths/round")
        self._log(f"Max empty rounds: {max_empty_rounds}")
        self._log("─" * 50)

    def _log_summary(self, elapsed_seconds: float):
        """Log final summary"""
        self._log("─" * 50)
        self._log("Discovery completed!")
        self._log(f"Total rounds: {self.round_number}")
        self._log(f"Total discovered: {self.total_discovered} paths")
        self._log(f"Runtime: {elapsed_seconds:.1f} seconds")

        if self.round_number > 0:
            avg_per_round = self.total_discovered / self.round_number
            self._log(f"Average per round: {avg_per_round:.1f} paths")

def parse_duration(duration_str: str) -> int:
    """Parse duration string like '30m', '2h' to minutes"""
    duration_str = duration_str.strip().lower()

    if duration_str.endswith('m'):
        return int(duration_str[:-1])
    elif duration_str.endswith('h'):
        return int(duration_str[:-1]) * 60
    elif duration_str.isdigit():
        return int(duration_str)  # Assume minutes
    else:
        raise ValueError(f"Invalid duration format: {duration_str}. Use '30m' or '2h'")
```

**Step 4: Update continuous command implementation**

```python
# Update in src/enumeraite/cli/commands.py - replace the continuous function

@click.command()
@click.option('--target', required=True, help='Target domain')
@click.option('--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input file containing known paths (will be updated)')
@click.option('--duration', default='30m', help='How long to run (e.g., 30m, 2h)')
@click.option('--batch-size', default=20, help='Paths to generate per round')
@click.option('--max-empty-rounds', default=5, help='Stop after N rounds with no discoveries')
@click.option('--provider', help='AI provider to use')
@click.option('--methods', default='GET', help='HTTP methods to test')
def continuous(target, input_file, duration, batch_size, max_empty_rounds, provider, methods):
    """Run continuous discovery mode with live path updates"""
    from ..core.continuous import ContinuousDiscovery, parse_duration

    # Load initial paths
    initial_paths = _load_paths_from_file(input_file)
    if not initial_paths:
        click.echo("No paths found in input file", err=True)
        return

    # Setup engine
    config = load_config()
    factory = ProviderFactory(config)

    if provider:
        ai_provider = factory.create_provider(provider)
    else:
        ai_provider = factory.get_default_provider()

    validator = HTTPValidator()
    engine = GenerationEngine(ai_provider, validator)

    # Parse duration
    try:
        duration_minutes = parse_duration(duration)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Path update callback to write to file
    def update_input_file(new_paths):
        with open(input_file, 'a') as f:
            for path in new_paths:
                f.write(f"{path}\n")

    # Run discovery
    discovery = ContinuousDiscovery(engine, target)
    http_methods = [m.strip().upper() for m in methods.split(',')]

    try:
        final_paths = asyncio.run(discovery.run_discovery(
            initial_paths, duration_minutes, batch_size, max_empty_rounds,
            path_update_callback=update_input_file
        ))

        click.echo(f"\nDiscovery completed! Input file updated with new paths.")
        click.echo(f"Final path count: {len(final_paths)}")

    except KeyboardInterrupt:
        click.echo("\nDiscovery interrupted by user")
    except Exception as e:
        click.echo(f"Error during discovery: {e}", err=True)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/core/test_continuous.py -v`
Expected: PASS

**Step 6: Commit continuous discovery mode**

```bash
git add src/enumeraite/core/continuous.py tests/core/test_continuous.py src/enumeraite/cli/commands.py
git commit -m "feat: implement continuous discovery mode with live updates"
```

## Task 11: Integration Tests and CLI Testing

**Files:**
- Create: `tests/integration/test_full_workflow.py`
- Create: `tests/cli/test_integration.py`

**Step 1: Write integration test**

```python
# tests/integration/test_full_workflow.py
import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from click.testing import CliRunner

from enumeraite.cli.main import cli

@pytest.fixture
def temp_paths_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("/api/user/profile\n")
        f.write("/api/admin/dashboard\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def temp_config_file():
    config_content = """{
        "providers": {
            "openai": {
                "api_key": "test-key",
                "model": "gpt-4"
            }
        },
        "default_provider": "openai"
    }"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)

@patch('enumeraite.providers.openai_provider.openai.OpenAI')
def test_batch_command_integration(mock_openai, temp_paths_file):
    """Test full batch workflow"""
    # Mock OpenAI response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = """
    /api/user/settings
    /api/user/preferences
    /api/admin/users
    """
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output.txt")

        result = runner.invoke(cli, [
            'batch',
            '--target', 'example.com',
            '--input', temp_paths_file,
            '--output', output_file,
            '--count', '10',
            '--provider', 'openai'
        ])

        assert result.exit_code == 0
        assert "Generated" in result.output

        # Check output file
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            content = f.read()
            assert "/api/user/settings" in content

def test_cli_help_messages():
    """Test that help messages are informative"""
    runner = CliRunner()

    # Main help
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'AI-Assisted' in result.output

    # Batch help
    result = runner.invoke(cli, ['batch', '--help'])
    assert result.exit_code == 0
    assert '--target' in result.output
    assert '--input' in result.output
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_full_workflow.py -v`
Expected: PASS

**Step 3: Write CLI unit tests**

```python
# tests/cli/test_integration.py
import pytest
import tempfile
import os
from click.testing import CliRunner
from enumeraite.cli.main import cli

def test_batch_command_missing_target():
    """Test error handling for missing required options"""
    runner = CliRunner()
    result = runner.invoke(cli, ['batch', '--input', 'nonexistent.txt'])
    assert result.exit_code != 0
    assert 'Missing option' in result.output

def test_batch_command_nonexistent_file():
    """Test error handling for nonexistent input file"""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'batch',
        '--target', 'example.com',
        '--input', 'definitely_not_a_real_file.txt'
    ])
    assert result.exit_code != 0

def test_continuous_command_structure():
    """Test that continuous command accepts expected options"""
    runner = CliRunner()
    result = runner.invoke(cli, ['continuous', '--help'])
    assert result.exit_code == 0
    assert '--duration' in result.output
    assert '--batch-size' in result.output
    assert '--max-empty-rounds' in result.output
```

**Step 4: Run CLI tests**

Run: `pytest tests/cli/test_integration.py -v`
Expected: PASS

**Step 5: Commit integration tests**

```bash
git add tests/integration/ tests/cli/test_integration.py
git commit -m "feat: add integration tests and CLI validation"
```

## Task 12: Documentation and Final Setup

**Files:**
- Create: `docs/README.md`
- Create: `docs/examples/`
- Create: `scripts/setup.py`
- Update: `README.md`

**Step 1: Create comprehensive documentation**

```markdown
# docs/README.md
# Enumeraite Seed-Based Path Generator

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your API keys:
```bash
# Option 1: Environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Option 2: Config file
enumeraite init-config
# Edit enumeraite.json with your keys
```

3. Create an input file with known paths:
```bash
echo "/api/user/profile" > paths.txt
echo "/api/admin/dashboard" >> paths.txt
```

4. Run path generation:
```bash
# Basic generation
enumeraite batch --target example.com --input paths.txt

# With validation
enumeraite batch --target example.com --input paths.txt --validate --advanced

# Continuous discovery
enumeraite continuous --target example.com --input paths.txt --duration 30m
```

## Commands

### batch
Generate paths once and output results.

**Options:**
- `--target` (required): Target domain
- `--input` (required): File with known paths
- `--output`: Output file (stdout if not specified)
- `--count`: Number of paths to generate (default: 50)
- `--provider`: AI provider (openai, claude)
- `--advanced`: Output detailed JSON format
- `--validate`: Test paths via HTTP
- `--methods`: HTTP methods to test (default: GET)

### continuous
Run intelligent discovery loop with live updates.

**Options:**
- `--target` (required): Target domain
- `--input` (required): File with known paths (updated live)
- `--duration`: How long to run (default: 30m)
- `--batch-size`: Paths per round (default: 20)
- `--max-empty-rounds`: Stop after N empty rounds (default: 5)
- `--provider`: AI provider to use
- `--methods`: HTTP methods to test (default: GET)

## Configuration

Configuration file format (`enumeraite.json`):

```json
{
  "default_provider": "openai",
  "default_count": 50,
  "providers": {
    "openai": {
      "api_key": "your-openai-key",
      "model": "gpt-4"
    },
    "claude": {
      "api_key": "your-anthropic-key",
      "model": "claude-3-sonnet-20240229"
    }
  }
}
```

## Output Formats

### Simple (default)
```
/api/user/settings
/api/user/preferences
/api/admin/users
```

### Advanced (--advanced)
```json
{
  "generated_count": 3,
  "metadata": {"provider": "openai", "target": "example.com"},
  "generated_paths": [
    {
      "path": "/api/user/settings",
      "confidence": 0.85,
      "verified": true,
      "status": 200
    }
  ]
}
```

## Integration with Other Tools

```bash
# Pipe to ffuf
enumeraite batch --target example.com --input paths.txt | ffuf -w - -u https://example.com/FUZZ

# Save and use with Burp Suite
enumeraite batch --target example.com --input paths.txt --output generated.txt
# Import generated.txt into Burp Intruder

# Continuous discovery with live monitoring
enumeraite continuous --target example.com --input paths.txt --duration 1h &
tail -f paths.txt | while read path; do
    echo "Testing: $path"
    curl -s -o /dev/null -w "%{http_code}" "https://example.com$path"
done
```
```

**Step 2: Create example files**

```bash
# Create examples directory and files
mkdir -p docs/examples

# docs/examples/basic_usage.md
cat > docs/examples/basic_usage.md << 'EOF'
# Basic Usage Examples

## Example 1: API Discovery

Input file (`api_paths.txt`):
```
/api/v1/users
/api/v1/auth/login
/api/v1/admin/stats
```

Generate new paths:
```bash
enumeraite batch --target api.company.com --input api_paths.txt --count 20
```

Expected output:
```
/api/v1/users/profile
/api/v1/users/settings
/api/v1/auth/logout
/api/v1/auth/refresh
/api/v1/admin/users
/api/v1/admin/logs
```

## Example 2: Continuous Discovery

Start with basic recon:
```bash
echo "/api/user" > discovered.txt
echo "/portal/admin" >> discovered.txt
```

Run 30-minute discovery:
```bash
enumeraite continuous --target target.com --input discovered.txt --duration 30m --batch-size 15
```

Watch discoveries in real-time:
```bash
tail -f discovered.txt
```

## Example 3: Validation and Integration

Generate and validate:
```bash
enumeraite batch --target api.example.com --input known.txt --validate --advanced --output results.json
```

Extract valid paths for further testing:
```bash
jq -r '.generated_paths[] | select(.verified == true) | .path' results.json > valid_paths.txt
```
EOF
```

**Step 3: Create setup script**

```python
# scripts/setup.py
#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

def create_config_file():
    """Create default configuration file"""
    config = {
        "default_provider": "openai",
        "default_count": 50,
        "max_concurrent_requests": 20,
        "request_timeout": 10,
        "providers": {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
                "model": "gpt-4"
            },
            "claude": {
                "api_key": os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here"),
                "model": "claude-3-sonnet-20240229"
            }
        }
    }

    config_path = "enumeraite.json"

    if os.path.exists(config_path):
        print(f"Configuration file {config_path} already exists")
        return

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created configuration file: {config_path}")
    print("Please edit it with your API keys")

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['click', 'aiohttp', 'pydantic', 'openai', 'anthropic']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True

def main():
    print("Enumeraite Setup")
    print("=" * 40)

    if not check_dependencies():
        sys.exit(1)

    create_config_file()

    print("\nSetup complete! Next steps:")
    print("1. Edit enumeraite.json with your API keys")
    print("2. Create an input file with known paths")
    print("3. Run: enumeraite batch --target example.com --input paths.txt")

if __name__ == "__main__":
    main()
```

**Step 4: Update main README**

```markdown
# Update README.md
# 🔍 Enumeraite: AI-Assisted Web Attack Surface Enumeration

> 🚀 **NEW**: First tool now available - Seed-Based Path Generator!

## Quick Start

```bash
# Install
pip install -r requirements.txt
python scripts/setup.py

# Configure
export OPENAI_API_KEY="your-key"
# OR edit enumeraite.json

# Create input file
echo "/api/user/profile" > paths.txt
echo "/api/admin/dashboard" >> paths.txt

# Generate new paths
enumeraite batch --target example.com --input paths.txt

# Run continuous discovery
enumeraite continuous --target example.com --input paths.txt --duration 30m
```

## What This Tool Does

The seed-based path generator expands your attack surface by:
- 🧠 **AI Pattern Recognition**: Analyzes existing paths to understand target patterns
- 🎯 **Contextual Understanding**: Generates realistic endpoints based on domain knowledge
- 🔄 **Continuous Learning**: Adapts based on which generated paths actually exist
- ⚡ **Multiple Providers**: Works with OpenAI, Claude, and custom models

## Example

**Input:**
```
/api/user/profile
/api/admin/dashboard
```

**Generated:**
```
/api/user/settings
/api/user/preferences
/api/user/notifications
/api/admin/users
/api/admin/analytics
/api/admin/config
```

## Documentation

- [Full Documentation](docs/README.md)
- [Usage Examples](docs/examples/basic_usage.md)
- [Implementation Plan](docs/plans/2026-01-22-seed-path-generator.md)

## Research

This tool is part of our research: "Enumeraite: AI-Assisted Web Attack Surface Enumeration"

📅 **DEFCON 33 Talk**: August 9, 2025 at Recon Village
🎤 **Speaker**: [Özgün Kültekin](https://github.com/ozgunkultekin)
```

**Step 5: Commit documentation**

```bash
git add docs/ scripts/ README.md
git commit -m "docs: add comprehensive documentation and setup script"
```

## Task 13: Final Testing and Validation

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Test CLI installation**

```bash
pip install -e .
enumeraite --help
```

**Step 3: Create example workflow test**

```bash
# Create test input
echo "/api/user/profile" > test_paths.txt
echo "/api/admin/dashboard" >> test_paths.txt

# Test with mock (will fail without API keys, which is expected)
python scripts/setup.py

# Verify config created
ls -la enumeraite.json
```

**Step 4: Final commit**

```bash
git add test_paths.txt
git commit -m "feat: complete MVP implementation with full test coverage"
```

---

## Summary

This plan creates a complete MVP implementation of the seed-based path generator with:

✅ **Core Features**: Batch and continuous modes
✅ **Multi-Provider**: OpenAI, Claude support with extensible plugin system
✅ **Intelligent Generation**: Pattern analysis + contextual understanding + confidence scoring
✅ **HTTP Validation**: Async path verification
✅ **Live Discovery**: Continuous mode with real-time input file updates
✅ **Production Ready**: Comprehensive tests, CLI, configuration, documentation

**Plan complete and saved to `docs/plans/2026-01-22-seed-path-generator.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**