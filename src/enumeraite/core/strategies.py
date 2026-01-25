"""Prompt strategies for different discovery types."""
from abc import ABC, abstractmethod
from typing import List, Tuple


class PromptStrategy(ABC):
    """Base class for prompt strategies."""

    @abstractmethod
    def build_prompts(self, known_items: List[str], target: str, count: int, max_depth: int = None) -> Tuple[str, str]:
        """Build system and user prompts for the AI model.

        Args:
            known_items: List of known items to learn from (paths or subdomains)
            target: Target domain or application
            count: Number of items to generate
            max_depth: Optional maximum depth for generated items

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        pass

    @abstractmethod
    def extract_items(self, content: str) -> List[str]:
        """Extract valid items from AI response content.

        Args:
            content: Raw response content from AI model

        Returns:
            List of extracted and validated items
        """
        pass

    @abstractmethod
    def is_valid_item(self, item: str) -> bool:
        """Validate that an item is well-formed and safe.

        Args:
            item: Item to validate (path or subdomain)

        Returns:
            True if valid, False otherwise
        """
        pass


class PathDiscoveryStrategy(PromptStrategy):
    """Strategy for API path discovery."""

    def build_prompts(self, known_items: List[str], target: str, count: int, max_depth: int = None) -> Tuple[str, str]:
        """Build prompts for API path generation."""

        system_prompt = """You are an expert web application security researcher specializing in API endpoint discovery and reconnaissance. Your task is to analyze known API paths and generate new, realistic endpoint possibilities for penetration testing and security assessment.

Key principles:
- Generate ACTUAL paths with real values, never use placeholders like {id}, {user_id}, or {slug}
- Use realistic identifiers like numbers (123, 456), common names (admin, user, test), or UUIDs
- Focus on discovering hidden or forgotten endpoints that might exist
- Consider REST patterns, naming conventions, and common web application structures
- Think like a developer who might have created additional endpoints beyond the obvious ones"""

        known_paths_str = "\n".join(known_items)
        user_prompt = f"""Analyze these known API paths for {target}:

{known_paths_str}

Generate {count} new potential API endpoint paths using:
1. Pattern analysis: Analyze REST conventions, naming patterns, and structural similarities, version patterns
2. Contextual understanding: Consider domain concepts, user roles, and common functionality
3. Common web application patterns: Think about typical CRUD operations, authentication, admin functions
4. Security focus: Consider endpoints that might be forgotten, debug paths, or admin interfaces
5. Lateral exploration: Discover similar endpoints in different versions (e.g., if /api/v1/users exists, try /api/v2/users)

Requirements:
- Generate ONLY the path portion starting with /
- Use actual values, NOT placeholders (e.g., /api/users/123 not /api/users/{{id}})
- One path per line, no explanations
- Focus on realistic, well-formed API paths that could actually exist
- Consider different user roles, resources, and operations
- Explore both deeper nesting AND lateral alternatives (same level, different versions/names)

Generated paths:"""

        return system_prompt, user_prompt

    def extract_items(self, content: str) -> List[str]:
        """Extract valid API paths from the response content."""
        items = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('/'):
                # Clean up the path - take first token in case of extra text
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
        return True


class SubdomainDiscoveryStrategy(PromptStrategy):
    """Strategy for subdomain discovery."""

    def build_prompts(self, known_items: List[str], target: str, count: int, max_depth: int = None) -> Tuple[str, str]:
        """Build prompts for subdomain generation."""

        system_prompt = """You are an expert at analyzing subdomain patterns and generating new potential subdomains for security testing. Your task is to analyze known subdomains and generate new, realistic possibilities based on the patterns you observe.

Key principles:
- Generate ACTUAL subdomain names, never use placeholders like {service} or {env}
- Focus on discovering hidden or forgotten subdomains that might exist
- Consider the naming patterns and conventions you see in the input
- Think about similar subdomains that developers might have created"""

        known_subdomains_str = "\n".join(known_items)

        user_prompt = f"""Analyze these known subdomains:

{known_subdomains_str}

Generate {count} new potential subdomains using:
1. Pattern analysis: Analyze naming conventions and structural similarities
2. Contextual understanding: Consider what these subdomains suggest about the organization
3. Lateral exploration: Discover similar subdomains with slight variations

Requirements:
- Generate full subdomain names (e.g., test.api.example.com)
- Use actual names, NOT placeholders
- One subdomain per line, no explanations
- Focus on realistic subdomains that could actually exist
- Consider variations of existing patterns

Generated subdomains:"""

        return system_prompt, user_prompt

    def extract_items(self, content: str) -> List[str]:
        """Extract valid subdomains from the response content."""
        items = []
        for line in content.split('\n'):
            line = line.strip()
            # Look for patterns that look like subdomains (contain dots, no slashes)
            if '.' in line and not line.startswith('/') and ' ' not in line:
                # Clean up the subdomain - take first token
                item = line.split()[0].lower()
                if self.is_valid_item(item):
                    items.append(item)
        return items

    def is_valid_item(self, item: str) -> bool:
        """Validate that a subdomain is well-formed and safe."""
        if not item:
            return False

        # Must contain at least one dot
        if '.' not in item:
            return False

        # Reject overly long subdomains
        if len(item) > 250:
            return False

        # Security checks
        if '//' in item or '..' in item or item.startswith('-') or item.endswith('-'):
            return False

        # Basic subdomain format check
        parts = item.split('.')
        if len(parts) < 2:
            return False

        # Each part should be valid
        for part in parts:
            if not part or len(part) > 63:  # DNS label length limit
                return False
            if not part.replace('-', '').replace('_', '').isalnum():
                return False

        return True


class PatternAnalysisStrategy(PromptStrategy):
    """Strategy for analyzing subdomain patterns and generating variants."""

    def build_prompts(self, known_items: List[str], target: str, count: int, max_depth: int = None) -> Tuple[str, str]:
        """Build prompts for pattern analysis of a single subdomain."""

        if len(known_items) != 1:
            raise ValueError("Pattern analysis requires exactly one subdomain input")

        subdomain = known_items[0]

        system_prompt = """You are an expert at analyzing complex subdomain naming patterns using AI reasoning. Your task is to deeply analyze a single subdomain, decompose it into meaningful components, understand the underlying pattern, and generate realistic variants.

You have extensive knowledge about:
- Technology naming conventions (AWS regions, product names, environments)
- Apple product ecosystem (iPhone, iPad, Mac, AirPods, Apple Watch, Apple TV, etc.)
- Cloud infrastructure patterns (use1, eu-west-1, ap-southeast, etc.)
- Service naming patterns (api, admin, staging, prod, dev, test, etc.)
- Instance/server identifiers (cx01, dx02, web1, db2, etc.)

Your response must be valid JSON with this exact structure:
{
  "decomposition": [
    {
      "value": "component_text",
      "type": "component_category",
      "description": "explanation of what this component represents",
      "alternatives": ["alt1", "alt2", "alt3"]
    }
  ],
  "reasoning": "detailed explanation of how you decomposed and understood this pattern",
  "pattern_template": "abstract representation like <action><product>-<env><digit>-<prefix><padded_digit>",
  "generated_variants": ["variant1.example.com", "variant2.example.com"],
  "confidence_score": 0.85
}"""

        user_prompt = f"""Analyze this subdomain and decompose its naming pattern: {subdomain}

Please:
1. Break down the subdomain into meaningful components (prefixes, product names, environment indicators, instance numbers, etc.)
2. Identify what each component represents and provide alternatives that follow the same pattern
3. Explain your reasoning for the decomposition
4. Create an abstract template showing the pattern structure
5. Generate {count} realistic variants using your understanding

Focus on:
- Recognizing product names (especially Apple products like iPhone, iPad, Mac)
- Environment/region indicators (use1, eu1, ap2, etc.)
- Service types (activate, enable, start, init, etc.)
- Instance/server identifiers with proper formatting (cx01, cx02, dx01, etc.)
- Maintaining realistic subdomain structure and DNS compliance

Generate variants that are plausible subdomains someone might actually use."""

        return system_prompt, user_prompt

    def extract_items(self, content: str) -> List[str]:
        """Extract generated variants from JSON response."""
        try:
            import json
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
            variants = response.get('generated_variants', [])

            # Validate each variant
            valid_variants = []
            for variant in variants:
                if isinstance(variant, str) and self.is_valid_item(variant):
                    valid_variants.append(variant)

            return valid_variants
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: try to extract subdomains line by line
            items = []
            for line in content.split('\n'):
                line = line.strip()
                if '.' in line and not line.startswith('/') and ' ' not in line:
                    item = line.split()[0].lower()
                    if self.is_valid_item(item):
                        items.append(item)
            return items

    def is_valid_item(self, item: str) -> bool:
        """Validate that a subdomain variant is well-formed."""
        if not item:
            return False

        # Must contain at least one dot
        if '.' not in item:
            return False

        # Reject overly long subdomains
        if len(item) > 250:
            return False

        # Security checks
        if '//' in item or '..' in item or item.startswith('-') or item.endswith('-'):
            return False

        # Basic subdomain format check
        parts = item.split('.')
        if len(parts) < 2:
            return False

        # Each part should be valid
        for part in parts:
            if not part or len(part) > 63:  # DNS label length limit
                return False
            if not part.replace('-', '').replace('_', '').isalnum():
                return False

        return True