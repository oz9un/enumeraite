"""Claude provider for AI path generation."""
import anthropic
from typing import List
from ..core.provider import BaseProvider
from ..core.models import GenerationResult

class ClaudeProvider(BaseProvider):
    """Claude-based provider for generating API paths."""

    def __init__(self, config):
        """Initialize Claude provider with API configuration."""
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.get("api_key"))
        self.model = config.get("model", "claude-3-sonnet-20240229")

    def generate_paths(self, known_paths: List[str], target: str, count: int) -> GenerationResult:
        """Generate new API paths using Claude's language model."""
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
        """Build the prompt for Claude API."""
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
        """Extract valid API paths from the response content."""
        paths = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('/'):
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
        return "claude"