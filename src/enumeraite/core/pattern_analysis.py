"""Pattern analysis engine for decomposing subdomain naming patterns."""

import json
import math

from .models import PatternAnalysisResult, PatternComponent
from .prompt_spec import SUBDOMAIN_SPEC
from .provider import BaseProvider
from .strategies import PatternAnalysisStrategy
from .text_safety import safe_exception_text, safe_model_text


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
        if not isinstance(subdomain, str):
            raise ValueError("subdomain must be a valid fully-qualified DNS name")
        subdomain = subdomain.strip().lower().rstrip(".")
        if not SUBDOMAIN_SPEC.validator(subdomain):
            raise ValueError("subdomain must be a valid fully-qualified DNS name")
        if isinstance(variant_count, bool) or not isinstance(variant_count, int):
            raise ValueError("variant_count must be an integer")
        if not 1 <= variant_count <= 5_000:
            raise ValueError("variant_count must be between 1 and 5000")
        # Use the strategy to build prompts
        system_prompt, user_prompt = self.strategy.build_prompts(
            known_items=[subdomain], target="", count=variant_count
        )

        # Get the AI response
        response = str(self.provider._make_api_call(system_prompt, user_prompt) or "")[:500_000]

        # Parse the JSON response (handle markdown code blocks)
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                # Extract JSON from markdown code block
                lines = cleaned_response.split("\n")
                start_idx = 0
                end_idx = len(lines)

                # Find start and end of JSON block
                for i, line in enumerate(lines):
                    if line.strip() == "```json":
                        start_idx = i + 1
                    elif line.strip() == "```" and i > start_idx:
                        end_idx = i
                        break

                cleaned_response = "\n".join(lines[start_idx:end_idx])
            elif cleaned_response.startswith("```"):
                # Handle generic code blocks
                lines = cleaned_response.split("\n")[1:-1]  # Remove first and last lines
                cleaned_response = "\n".join(lines)

            analysis_data = json.loads(cleaned_response)
            if not isinstance(analysis_data, dict):
                raise TypeError("analysis response must be a JSON object")

            # Treat every model-provided field as untrusted structured data.
            components = []
            decomposition = analysis_data.get("decomposition", [])
            if not isinstance(decomposition, list):
                decomposition = []
            for comp_data in decomposition[:100]:
                if not isinstance(comp_data, dict):
                    continue
                alternatives = comp_data.get("alternatives", [])
                if not isinstance(alternatives, list):
                    alternatives = []
                components.append(
                    PatternComponent(
                        value=safe_model_text(comp_data.get("value", ""), 200),
                        type=safe_model_text(comp_data.get("type", ""), 100),
                        description=safe_model_text(comp_data.get("description", ""), 1000),
                        alternatives=[safe_model_text(value, 200) for value in alternatives[:20]],
                    )
                )

            raw_variants = analysis_data.get("generated_variants", [])
            if not isinstance(raw_variants, list):
                raw_variants = []
            valid_variants = _scoped_variants(raw_variants, subdomain, variant_count, self.strategy)

            # Create the result object
            result = PatternAnalysisResult(
                original_subdomain=subdomain,
                decomposition=components,
                reasoning=safe_model_text(analysis_data.get("reasoning", ""), 4000, multiline=True),
                pattern_template=safe_model_text(analysis_data.get("pattern_template", ""), 1000),
                generated_variants=valid_variants,
                confidence_score=_clamp_confidence(analysis_data.get("confidence_score", 0.0)),
                metadata={
                    "provider": safe_model_text(self.provider.get_provider_name(), 200),
                    "model": safe_model_text(getattr(self.provider, "model", "unknown"), 200),
                    "requested_variants": variant_count,
                    "actual_variants": len(valid_variants),
                },
            )

            return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback: try to create a basic result with extracted variants
            fallback_variants = _scoped_variants(
                self.strategy.extract_items(response),
                subdomain,
                variant_count,
                self.strategy,
            )

            return PatternAnalysisResult(
                original_subdomain=subdomain,
                decomposition=[],
                reasoning=safe_model_text(
                    "Failed to parse structured response: "
                    f"{safe_exception_text(e, 500)}. Raw response: "
                    f"{safe_exception_text(response[:200], 200)}...",
                    1000,
                    multiline=True,
                ),
                pattern_template="<unknown_pattern>",
                generated_variants=fallback_variants,
                confidence_score=0.1,
                metadata={
                    "provider": safe_model_text(self.provider.get_provider_name(), 200),
                    "model": safe_model_text(getattr(self.provider, "model", "unknown"), 200),
                    "requested_variants": variant_count,
                    "actual_variants": len(fallback_variants),
                    "parse_error": safe_exception_text(e, 500),
                },
            )

    def _extract_subdomain_base(self, subdomain: str) -> str:
        """Extract the base part of a subdomain for pattern analysis.

        For example: 'activateiphone-use1-cx02.example.com' -> 'activateiphone-use1-cx02'

        Args:
            subdomain: Full subdomain with domain

        Returns:
            Base subdomain pattern without the domain suffix
        """
        if "." in subdomain:
            # Take everything before the last two parts (assuming format like sub.example.com)
            parts = subdomain.split(".")
            if len(parts) >= 3:
                return ".".join(parts[:-2])
            elif len(parts) == 2:
                return parts[0]
        return subdomain


def _scoped_variants(values, subdomain, limit, strategy):
    """Normalize, deduplicate, and keep variants under the input's parent."""
    normalized_input = subdomain.lower().rstrip(".")
    parts = normalized_input.split(".")
    parent = ".".join(parts[1:]) if len(parts) >= 2 else ""
    seen = {SUBDOMAIN_SPEC.identity(subdomain)}
    result = []
    for variant in values:
        if not isinstance(variant, str):
            continue
        value = variant.strip().lower().rstrip(".")
        identity = SUBDOMAIN_SPEC.identity(value)
        if (
            parent
            and value.endswith("." + parent)
            and strategy.is_valid_item(value)
            and identity not in seen
        ):
            seen.add(identity)
            result.append(value)
        if len(result) >= limit:
            break
    return result


def _clamp_confidence(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return min(1.0, max(0.0, score))
