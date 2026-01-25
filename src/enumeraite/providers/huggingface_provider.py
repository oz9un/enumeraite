"""HuggingFace provider for custom enumeraite models."""
import re
from typing import List, Optional
from ..core.provider import BaseProvider
from ..core.models import GenerationResult
from ..core.strategies import PromptStrategy

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class HuggingFaceProvider(BaseProvider):
    """HuggingFace-based provider using custom enumeraite models."""

    def __init__(self, config):
        """Initialize HuggingFace provider with model configuration."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers library is not installed. Install with: pip install transformers torch")

        super().__init__(config)
        # Model will be set by CLI based on generation type if not specified
        self.model_name = config.get("model", "enumeraite/Enumeraite-x-Qwen3-4B-Path")
        self.device = config.get("device", "auto")
        self.max_length = config.get("max_length", 512)
        self.temperature = config.get("temperature", 0.6)  # Slightly more creative
        self.do_sample = config.get("do_sample", True)
        self.max_new_tokens = config.get("max_new_tokens", 250)  # Controlled length
        self.top_k = config.get("top_k", 50)  # Add top_k for better quality

        # Load model and tokenizer
        print(f"Loading HuggingFace model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32,
            trust_remote_code=True
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_paths(self, known_paths: List[str], target: str, count: int, max_depth: int = None, strategy: PromptStrategy = None) -> GenerationResult:
        """Generate new paths using custom enumeraite models."""
        # Force custom prompts for enumeraite models (they're fine-tuned for simpler formats)
        if "enumeraite" in self.model_name.lower():
            full_prompt = self._build_enumeraite_prompt(known_paths, target, count, max_depth)
        elif strategy:
            system_prompt, user_prompt = strategy.build_prompts(known_paths, target, count, max_depth)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            # Use enumeraite-specific prompting for custom models
            full_prompt = self._build_enumeraite_prompt(known_paths, target, count, max_depth)

        # Tokenize input
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - 200,  # Leave room for generation
            padding=True
        )

        # Move inputs to same device as model (handles CUDA, MPS, CPU)
        if hasattr(self.model, 'device') and self.model.device.type != 'cpu':
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=0.9,
                top_k=self.top_k,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract generated content
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new content (remove the input prompt)
        if full_prompt in generated_text:
            generated_content = generated_text[len(full_prompt):].strip()
        else:
            generated_content = generated_text.strip()

        # Extract paths from generated content
        # Always use custom extraction for enumeraite models (they output in different format)
        if "enumeraite" in self.model_name.lower():
            paths = self._extract_paths(generated_content)[:count]
        elif strategy:
            paths = strategy.extract_items(generated_content)[:count]
        else:
            paths = self._extract_paths(generated_content)[:count]

        confidence_scores = self._calculate_confidence_scores(paths, known_paths)

        # Metadata for tracking
        metadata = {
            "provider": "huggingface",
            "model": self.model_name,
            "target": target,
            "generated_length": len(generated_content),
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown"
        }

        return GenerationResult(
            paths=paths,
            confidence_scores=confidence_scores,
            metadata=metadata
        )

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make an API call to HuggingFace model and return the response content.

        This method is used by pattern analysis and other components that need
        direct access to the AI response without path extraction.

        Args:
            system_prompt: System prompt for the AI
            user_prompt: User prompt for the AI

        Returns:
            Raw response content from HuggingFace model
        """
        # Combine system and user prompts for HuggingFace models
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Tokenize input
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - 200,
            padding=True
        )

        # Move inputs to same device as model (handles CUDA, MPS, CPU)
        if hasattr(self.model, 'device') and self.model.device.type != 'cpu':
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=0.9,
                top_k=self.top_k,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract generated content
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new content (remove the input prompt)
        if full_prompt in generated_text:
            generated_content = generated_text[len(full_prompt):].strip()
        else:
            generated_content = generated_text.strip()

        return generated_content

    def _build_enumeraite_prompt(self, known_paths: List[str], target: str, count: int, max_depth: int = None) -> str:
        """Build optimized prompt for enumeraite custom models."""
        # Detect if we're working with subdomains or paths
        is_subdomain = any("." in path and not path.startswith("/") for path in known_paths)

        if is_subdomain:
            return self._build_subdomain_prompt(known_paths, target, count)
        else:
            return self._build_path_prompt(known_paths, target, count, max_depth)

    def _build_subdomain_prompt(self, known_subdomains: List[str], target: str, count: int) -> str:
        """Build prompt optimized for subdomain generation."""
        known_str = "\n".join(known_subdomains)

        prompt = f"""Generate {count} new potential subdomains based on the patterns in these known subdomains:

Known subdomains:
{known_str}

Analyze the naming patterns, conventions, and structure. Generate realistic variants that follow similar patterns.

New subdomains:
"""
        return prompt

    def _build_path_prompt(self, known_paths: List[str], target: str, count: int, max_depth: int = None) -> str:
        """Build prompt optimized for API path generation."""
        known_str = "\n".join(known_paths)

        prompt = f"""Generate {count} new API endpoint paths based on these known paths:

Known paths:
{known_str}

Analyze REST patterns, naming conventions, and API structure. Generate realistic endpoints that could exist.

New paths:
"""
        return prompt

    def _extract_paths(self, content: str) -> List[str]:
        """Extract valid paths or subdomains from generated content."""
        items = []
        for line in content.split('\n'):
            line = line.strip()

            # Skip empty lines or lines that look like comments
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            # Extract the first token (in case of extra explanation)
            item = line.split()[0] if line.split() else ""

            if self._is_valid_item(item):
                items.append(item)

        return items

    def _is_valid_item(self, item: str) -> bool:
        """Validate if item is a proper path or subdomain."""
        if not item:
            return False

        # Check if it's a path (starts with /)
        if item.startswith('/'):
            return self._is_valid_path(item)

        # Check if it's a subdomain (contains dots)
        if '.' in item:
            return self._is_valid_subdomain(item)

        return False

    def _is_valid_path(self, path: str) -> bool:
        """Validate API path format."""
        if not path.startswith('/'):
            return False
        if len(path) > 200:
            return False
        if '..' in path or '//' in path:
            return False
        return True

    def _is_valid_subdomain(self, subdomain: str) -> bool:
        """Validate subdomain format."""
        if len(subdomain) > 253:
            return False
        if subdomain.startswith('.') or subdomain.endswith('.'):
            return False
        if '..' in subdomain:
            return False
        return True

    def _calculate_confidence_scores(self, items: List[str], known_items: List[str]) -> List[float]:
        """Calculate confidence scores for generated items."""
        scores = []
        for item in items:
            score = 0.6  # Base score for custom model outputs

            # Boost score for pattern similarity
            for known_item in known_items:
                similarity = self._calculate_similarity(item, known_item)
                score = max(score, 0.4 + similarity * 0.4)

            # Boost for common patterns
            if any(pattern in item.lower() for pattern in ['api', 'admin', 'user', 'auth', 'test', 'dev']):
                score += 0.1

            scores.append(min(score, 1.0))

        return scores

    def _calculate_similarity(self, item1: str, item2: str) -> float:
        """Calculate similarity between two items."""
        # For paths, compare segments
        if item1.startswith('/') and item2.startswith('/'):
            segments1 = set(item1.split('/')[1:])
            segments2 = set(item2.split('/')[1:])
        # For subdomains, compare parts
        elif '.' in item1 and '.' in item2:
            segments1 = set(item1.split('.'))
            segments2 = set(item2.split('.'))
        else:
            return 0.0

        if not segments1 or not segments2:
            return 0.0

        intersection = segments1.intersection(segments2)
        union = segments1.union(segments2)
        return len(intersection) / len(union) if union else 0.0

    def get_provider_name(self) -> str:
        """Return the provider name for identification."""
        return "huggingface"