"""Path function analysis engine for generating function-specific API endpoints."""
import json
from typing import List, Tuple
from .provider import BaseProvider
from .models import PathFunctionAnalysis, PathComponent
from .strategies import PromptStrategy


class PathFunctionAnalysisStrategy(PromptStrategy):
    """Strategy for analyzing API paths and generating function-specific variants."""

    def build_prompts(self, known_items: List[str], target: str, count: int, max_depth: int = None) -> Tuple[str, str]:
        """Build prompts for path function analysis.

        Expected format: known_items[0] = path, known_items[1] = function_context
        """
        if len(known_items) != 2:
            raise ValueError("Path function analysis requires exactly two inputs: path and function context")

        path = known_items[0]
        function_context = known_items[1]

        system_prompt = """You are an expert at analyzing API endpoint patterns and generating function-specific path variants for security testing and reconnaissance. Your task is to deeply understand a given API path, analyze its structure and naming conventions, and generate realistic path variants that implement the requested functionality.

You have extensive knowledge about:
- REST API patterns and conventions (CRUD operations, resource naming)
- Common API frameworks (Express, Django, Spring Boot, Rails, Flask)
- API versioning patterns (v1, v2, api/v1, etc.)
- Resource naming conventions (users, usr, user, etc.)
- Action naming patterns (create/crt, delete/del/dlt/rmv, update/upd, etc.)
- HTTP method implications and path structures
- Admin/privileged endpoint patterns
- Batch operation patterns
- API authentication and authorization patterns

Your response must be valid JSON with this exact structure:
{
  "path_breakdown": [
    {
      "value": "component_text",
      "type": "component_type",
      "description": "what this component represents",
      "position": 0
    }
  ],
  "function_analysis": "detailed analysis of how the requested function would be implemented",
  "reasoning": "explanation of your path generation logic and patterns identified",
  "generated_paths": ["path1", "path2", "path3"],
  "confidence_score": 0.85
}

Component types should be: "api_prefix", "version", "resource", "action", "parameter", "namespace", "identifier"."""

        user_prompt = f"""Analyze this API path and generate variants for the requested functionality:

Path: {path}
Requested Functionality: {function_context}

Please:
1. Break down the path into meaningful components (API prefix, version, resource, action, etc.)
2. Analyze what the original path does and how it's structured
3. Understand the requested functionality and how it would typically be implemented
4. Generate {count} realistic API path variants that implement the requested functionality
5. Consider common naming patterns, abbreviations, and alternative implementations

Focus on:
- Understanding REST conventions and resource patterns
- Recognizing common action abbreviations (crt→create, dlt→delete, upd→update)
- Considering version variations (v1, v2, etc.)
- Thinking about different implementation approaches (usr vs user vs users)
- Maintaining realistic API path structure and conventions
- Generating paths that a developer would actually implement

Generate paths that are plausible endpoints someone might discover during API reconnaissance."""

        return system_prompt, user_prompt

    def extract_items(self, content: str) -> List[str]:
        """Extract generated paths from JSON response."""
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_response = content.strip()
            if cleaned_response.startswith('```json'):
                # Extract JSON from markdown code block
                lines = cleaned_response.split('\n')
                start_idx = 0
                end_idx = len(lines)

                # Find start and end of JSON block
                for i, line in enumerate(lines):
                    if line.strip() == '```json':
                        start_idx = i + 1
                    elif line.strip() == '```' and i > start_idx:
                        end_idx = i
                        break

                cleaned_response = '\n'.join(lines[start_idx:end_idx])
            elif cleaned_response.startswith('```'):
                # Handle generic code blocks
                lines = cleaned_response.split('\n')[1:-1]  # Remove first and last lines
                cleaned_response = '\n'.join(lines)

            response = json.loads(cleaned_response)
            paths = response.get('generated_paths', [])

            # Validate each path
            valid_paths = []
            for path in paths:
                if isinstance(path, str) and self.is_valid_item(path):
                    valid_paths.append(path)

            return valid_paths
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: try to extract paths line by line
            items = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('/') and ' ' not in line:
                    item = line.split()[0]
                    if self.is_valid_item(item):
                        items.append(item)
            return items

    def is_valid_item(self, item: str) -> bool:
        """Validate that a path is well-formed and safe."""
        if not item.startswith('/'):
            return False
        if len(item) > 200:  # Reject overly long paths
            return False
        if '..' in item or '//' in item:  # Security check
            return False
        # Basic path format checks
        if item.count('/') > 10:  # Reject overly deep paths
            return False
        return True


class PathFunctionAnalysisEngine:
    """Engine for analyzing API paths and generating function-specific variants."""

    def __init__(self, provider: BaseProvider):
        """Initialize the path function analysis engine.

        Args:
            provider: AI provider for path analysis
        """
        self.provider = provider
        self.strategy = PathFunctionAnalysisStrategy()

    def analyze_path_function(self, path: str, function_context: str, variant_count: int = 20) -> PathFunctionAnalysis:
        """Analyze a path and generate function-specific variants.

        Args:
            path: The API path to analyze (e.g., '/api/v1/usr_crt')
            function_context: The requested functionality (e.g., 'user deletion')
            variant_count: Number of variants to generate

        Returns:
            PathFunctionAnalysis with breakdown, analysis, and generated paths
        """
        # Use the strategy to build prompts
        system_prompt, user_prompt = self.strategy.build_prompts(
            known_items=[path, function_context],
            target="",
            count=variant_count
        )

        # Get the AI response
        response = self.provider._make_api_call(system_prompt, user_prompt)

        # Parse the JSON response (handle markdown code blocks)
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                # Extract JSON from markdown code block
                lines = cleaned_response.split('\n')
                start_idx = 0
                end_idx = len(lines)

                # Find start and end of JSON block
                for i, line in enumerate(lines):
                    if line.strip() == '```json':
                        start_idx = i + 1
                    elif line.strip() == '```' and i > start_idx:
                        end_idx = i
                        break

                cleaned_response = '\n'.join(lines[start_idx:end_idx])
            elif cleaned_response.startswith('```'):
                # Handle generic code blocks
                lines = cleaned_response.split('\n')[1:-1]  # Remove first and last lines
                cleaned_response = '\n'.join(lines)

            analysis_data = json.loads(cleaned_response)

            # Convert path breakdown data to PathComponent objects
            components = []
            for comp_data in analysis_data.get('path_breakdown', []):
                component = PathComponent(
                    value=comp_data.get('value', ''),
                    type=comp_data.get('type', ''),
                    description=comp_data.get('description', ''),
                    position=comp_data.get('position', 0)
                )
                components.append(component)

            # Extract and validate generated paths
            raw_paths = analysis_data.get('generated_paths', [])
            valid_paths = []
            for path_item in raw_paths:
                if isinstance(path_item, str) and self.strategy.is_valid_item(path_item):
                    valid_paths.append(path_item)

            # Create the result object
            result = PathFunctionAnalysis(
                original_path=path,
                function_context=function_context,
                path_breakdown=components,
                function_analysis=analysis_data.get('function_analysis', ''),
                reasoning=analysis_data.get('reasoning', ''),
                generated_paths=valid_paths,
                confidence_score=analysis_data.get('confidence_score', 0.0),
                metadata={
                    'provider': self.provider.get_provider_name(),
                    'model': getattr(self.provider, 'model', 'unknown'),
                    'requested_variants': variant_count,
                    'actual_variants': len(valid_paths)
                }
            )

            return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback: try to create a basic result with extracted paths
            fallback_paths = self.strategy.extract_items(response)

            return PathFunctionAnalysis(
                original_path=path,
                function_context=function_context,
                path_breakdown=[],
                function_analysis=f"Failed to parse structured response: {str(e)}",
                reasoning=f"Raw response: {response[:200]}...",
                generated_paths=fallback_paths,
                confidence_score=0.1,
                metadata={
                    'provider': self.provider.get_provider_name(),
                    'model': getattr(self.provider, 'model', 'unknown'),
                    'requested_variants': variant_count,
                    'actual_variants': len(fallback_paths),
                    'parse_error': str(e)
                }
            )