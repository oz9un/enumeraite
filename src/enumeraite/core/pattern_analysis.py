"""Pattern analysis engine for decomposing subdomain naming patterns."""
import json
from typing import List
from .provider import BaseProvider
from .models import PatternAnalysisResult, PatternComponent
from .strategies import PatternAnalysisStrategy


class PatternAnalysisEngine:
    """Engine for analyzing subdomain patterns using AI reasoning."""

    def __init__(self, provider: BaseProvider):
        """Initialize the pattern analysis engine.

        Args:
            provider: AI provider for pattern analysis
        """
        self.provider = provider
        self.strategy = PatternAnalysisStrategy()

    def analyze_pattern(self, subdomain: str, variant_count: int = 20) -> PatternAnalysisResult:
        """Analyze a subdomain pattern and generate variants.

        Args:
            subdomain: The subdomain to analyze (e.g., 'activateiphone-use1-cx02.example.com')
            variant_count: Number of variants to generate

        Returns:
            PatternAnalysisResult with decomposition, reasoning, and generated variants
        """
        # Use the strategy to build prompts
        system_prompt, user_prompt = self.strategy.build_prompts(
            known_items=[subdomain],
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

            # Convert decomposition data to PatternComponent objects
            components = []
            for comp_data in analysis_data.get('decomposition', []):
                component = PatternComponent(
                    value=comp_data.get('value', ''),
                    type=comp_data.get('type', ''),
                    description=comp_data.get('description', ''),
                    alternatives=comp_data.get('alternatives', [])
                )
                components.append(component)

            # Extract and validate generated variants
            raw_variants = analysis_data.get('generated_variants', [])
            valid_variants = []
            for variant in raw_variants:
                if isinstance(variant, str) and self.strategy.is_valid_item(variant):
                    valid_variants.append(variant)

            # Create the result object
            result = PatternAnalysisResult(
                original_subdomain=subdomain,
                decomposition=components,
                reasoning=analysis_data.get('reasoning', ''),
                pattern_template=analysis_data.get('pattern_template', ''),
                generated_variants=valid_variants,
                confidence_score=analysis_data.get('confidence_score', 0.0),
                metadata={
                    'provider': self.provider.get_provider_name(),
                    'model': getattr(self.provider, 'model', 'unknown'),
                    'requested_variants': variant_count,
                    'actual_variants': len(valid_variants)
                }
            )

            return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback: try to create a basic result with extracted variants
            fallback_variants = self.strategy.extract_items(response)

            return PatternAnalysisResult(
                original_subdomain=subdomain,
                decomposition=[],
                reasoning=f"Failed to parse structured response: {str(e)}. Raw response: {response[:200]}...",
                pattern_template="<unknown_pattern>",
                generated_variants=fallback_variants,
                confidence_score=0.1,
                metadata={
                    'provider': self.provider.get_provider_name(),
                    'model': getattr(self.provider, 'model', 'unknown'),
                    'requested_variants': variant_count,
                    'actual_variants': len(fallback_variants),
                    'parse_error': str(e)
                }
            )

    def _extract_subdomain_base(self, subdomain: str) -> str:
        """Extract the base part of a subdomain for pattern analysis.

        For example: 'activateiphone-use1-cx02.example.com' -> 'activateiphone-use1-cx02'

        Args:
            subdomain: Full subdomain with domain

        Returns:
            Base subdomain pattern without the domain suffix
        """
        if '.' in subdomain:
            # Take everything before the last two parts (assuming format like sub.example.com)
            parts = subdomain.split('.')
            if len(parts) >= 3:
                return '.'.join(parts[:-2])
            elif len(parts) == 2:
                return parts[0]
        return subdomain