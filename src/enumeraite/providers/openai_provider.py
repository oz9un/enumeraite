"""OpenAI provider for AI path generation."""
import re
import openai
from typing import List
from ..core.provider import BaseProvider
from ..core.models import GenerationResult

class OpenAIProvider(BaseProvider):
    """OpenAI-based provider for generating API paths."""

    def __init__(self, config):
        """Initialize OpenAI provider with API configuration."""
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config.get("api_key"))
        self.model = config.get("model", "gpt-4")

    def generate_paths(self, known_paths: List[str], target: str, count: int) -> GenerationResult:
        """Generate new API paths using OpenAI's language model."""
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
        """Build the prompt for OpenAI API."""
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
        """Extract valid API paths from the response content."""
        paths = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('/'):
                # Clean up the path - take first token in case of extra text
                path = line.split()[0]
                if self._is_valid_path(path):
                    paths.append(path)
        return paths

    def _is_valid_path(self, path: str) -> bool:
        """Validate that a path is well-formed and safe."""
        if not path.startswith('/'):
            return False
        if len(path) > 200:  # Reject overly long paths
            return False
        if '..' in path or '//' in path:  # Security check
            return False
        return True

    def _calculate_confidence_scores(self, paths: List[str], known_paths: List[str]) -> List[float]:
        """Calculate confidence scores for generated paths."""
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
        """Calculate similarity between two paths based on common segments."""
        segments1 = set(path1.split('/')[1:])  # Skip empty first element
        segments2 = set(path2.split('/')[1:])

        if not segments1 or not segments2:
            return 0.0

        intersection = segments1.intersection(segments2)
        union = segments1.union(segments2)
        return len(intersection) / len(union) if union else 0.0

    def get_provider_name(self) -> str:
        """Return the provider name for identification."""
        return "openai"