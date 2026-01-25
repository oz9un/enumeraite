"""Claude provider for AI path generation."""
import anthropic
from typing import List
from ..core.provider import BaseProvider
from ..core.models import GenerationResult
from ..core.strategies import PromptStrategy

class ClaudeProvider(BaseProvider):
    """Claude-based provider for generating API paths."""

    def __init__(self, config):
        """Initialize Claude provider with API configuration."""
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.get("api_key"))
        self.model = config.get("model", "anthropic/claude-sonnet-4")

    def generate_paths(self, known_paths: List[str], target: str, count: int, max_depth: int = None, strategy: PromptStrategy = None) -> GenerationResult:
        """Generate new API paths using Claude's language model."""
        if strategy:
            system_prompt, user_prompt = strategy.build_prompts(known_paths, target, count, max_depth)
        else:
            # Fallback to legacy prompts for backward compatibility
            system_prompt, user_prompt = self._build_prompts(known_paths, target, count, max_depth)

        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=2000,
            temperature=0.7
        )

        content = response.content[0].text
        if strategy:
            paths = strategy.extract_items(content)[:count]
        else:
            # Fallback to legacy extraction for backward compatibility
            paths = self._extract_paths(content)[:count]
        confidence_scores = self._calculate_confidence_scores(paths, known_paths)

        # Extract token usage information
        metadata = {
            "provider": "claude",
            "model": self.model,
            "target": target
        }

        # Add token usage if available
        if hasattr(response, 'usage') and response.usage:
            metadata["token_usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

        return GenerationResult(
            paths=paths,
            confidence_scores=confidence_scores,
            metadata=metadata
        )

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make an API call to Claude and return the response content.

        This method is used by pattern analysis and other components that need
        direct access to the AI response without path extraction.

        Args:
            system_prompt: System prompt for the AI
            user_prompt: User prompt for the AI

        Returns:
            Raw response content from Claude
        """
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4000,  # Higher limit for analysis tasks
            temperature=0.7
        )

        return response.content[0].text

    def _build_prompts(self, known_paths: List[str], target: str, count: int, max_depth: int = None) -> tuple[str, str]:
        """Build system and user prompts for Claude API."""

        system_prompt = """You are an expert web application security researcher specializing in API endpoint discovery and reconnaissance. Your task is to analyze known API paths and generate new, realistic endpoint possibilities for penetration testing and security assessment.

Key principles:
- Generate ACTUAL paths with real values, never use placeholders like {id}, {user_id}, or {slug}
- Use realistic identifiers like numbers (123, 456), common names (admin, user, test), or UUIDs
- Focus on discovering hidden or forgotten endpoints that might exist
- Consider REST patterns, naming conventions, and common web application structures
- Think like a developer who might have created additional endpoints beyond the obvious ones"""

        known_paths_str = "\n".join(known_paths)
        user_prompt = f"""Analyze these known API paths for {target}:

{known_paths_str}

Generate {count} new potential API endpoint paths using:
1. Pattern analysis - identify REST patterns, naming conventions, URL structures, version patterns
2. Contextual understanding - consider what functionality these paths suggest
3. Common patterns - think about typical CRUD operations, authentication, admin functions
4. Security focus - consider endpoints that might be forgotten, debug paths, or admin interfaces
5. Lateral exploration - discover similar endpoints in different versions (e.g., if /api/v1/users exists, try /api/v2/users)

Requirements:
- Return ONLY the path portion starting with /
- Use actual values, NOT placeholders (e.g., /api/users/123 not /api/users/{{id}})
- One path per line, no explanations
- Focus on realistic, well-formed paths that could actually exist
- Consider different user roles, resources, and operations
- Explore both deeper nesting AND lateral alternatives (same level, different versions/names)

Generated paths:"""

        return system_prompt, user_prompt

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