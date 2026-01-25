#!/usr/bin/env python3
"""
Example usage of Enumeraite for programmatic API endpoint discovery.

This script demonstrates how to use Enumeraite's core components
without the CLI interface.
"""

import asyncio
import os
from typing import List

from enumeraite.core.config import Config, load_config
from enumeraite.core.factory import ProviderFactory
from enumeraite.core.engine import GenerationEngine
from enumeraite.core.validator import HTTPValidator
from enumeraite.core.continuous import ContinuousDiscovery


async def basic_generation_example():
    """Basic path generation without validation."""
    print("=== Basic Generation Example ===")

    # Load configuration
    try:
        config = load_config()
        factory = ProviderFactory(config)
        provider = factory.get_default_provider()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please ensure you have API keys configured.")
        return

    # Create engine without validator
    engine = GenerationEngine(provider)

    # Known paths
    known_paths = [
        "/api/users",
        "/api/auth/login",
        "/admin/dashboard"
    ]

    # Generate new paths
    result = engine.generate_paths(known_paths, "example.com", 10)

    print(f"Generated {len(result.paths)} paths:")
    for i, (path, confidence) in enumerate(zip(result.paths, result.confidence_scores)):
        print(f"  {i+1:2d}. {path} (confidence: {confidence:.3f})")

    print(f"\nMetadata: {result.metadata}")


async def validation_example():
    """Path generation with HTTP validation."""
    print("\n=== Validation Example ===")

    # Load configuration
    try:
        config = load_config()
        factory = ProviderFactory(config)
        provider = factory.get_default_provider()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Create engine with validator
    validator = HTTPValidator()
    engine = GenerationEngine(provider, validator)

    # Known paths
    known_paths = [
        "/api/users",
        "/api/auth/login"
    ]

    # Generate and validate paths
    generation_result, validation_results = await engine.generate_and_validate_paths(
        known_paths, "httpbin.org", 5, ["GET"]
    )

    print(f"Generated and tested {len(generation_result.paths)} paths:")

    # Create validation lookup
    validation_lookup = {}
    for val_result in validation_results:
        # Extract path from full URL
        if val_result.path.startswith('http'):
            path_parts = val_result.path.split('/', 3)
            if len(path_parts) > 3:
                path = '/' + path_parts[3]
            else:
                path = '/'
        else:
            path = val_result.path
        validation_lookup[path] = val_result

    for i, (path, confidence) in enumerate(zip(generation_result.paths, generation_result.confidence_scores)):
        validation = validation_lookup.get(path)
        status = "✓" if validation and validation.exists else "✗"
        status_code = validation.status_code if validation else "N/A"

        print(f"  {i+1:2d}. {status} {path} (confidence: {confidence:.3f}, status: {status_code})")


async def continuous_discovery_example():
    """Continuous discovery example."""
    print("\n=== Continuous Discovery Example ===")

    # Load configuration
    try:
        config = load_config()
        factory = ProviderFactory(config)
        provider = factory.get_default_provider()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Create engine with validator
    validator = HTTPValidator()
    engine = GenerationEngine(provider, validator)

    # Create continuous discovery instance
    discovery = ContinuousDiscovery(engine, "httpbin.org")

    # Known paths
    known_paths = [
        "/status/200",
        "/get",
        "/post"
    ]

    # Callback functions
    def progress_callback(round_num: int, new_count: int, total: int) -> None:
        if new_count > 0:
            print(f"  Round {round_num}: Found {new_count} new paths (total: {total})")

    discovered_paths = []
    def path_update_callback(new_paths: List[str]) -> None:
        discovered_paths.extend(new_paths)
        for path in new_paths:
            print(f"    ✓ Discovered: {path}")

    print("Running continuous discovery for 1 minute...")
    print("(This will make real HTTP requests to httpbin.org)")

    try:
        # Run for a short time with small batches
        final_paths = await discovery.run_discovery(
            known_paths,
            duration_minutes=0.1,  # 6 seconds
            batch_size=3,
            max_empty_rounds=2,
            progress_callback=progress_callback,
            path_update_callback=path_update_callback
        )

        print(f"\nDiscovery completed!")
        print(f"Total paths after discovery: {len(final_paths)}")
        print(f"New paths discovered: {len(discovered_paths)}")

    except Exception as e:
        print(f"Discovery error: {e}")


def manual_provider_example():
    """Example of manually creating providers."""
    print("\n=== Manual Provider Example ===")

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY not found in environment variables")
        return

    # Create config manually
    config_data = {
        "providers": {
            "openai": {
                "api_key": openai_key,
                "model": "gpt-3.5-turbo",
                "max_tokens": 500
            }
        },
        "default_provider": "openai"
    }

    config = Config.from_dict(config_data)
    factory = ProviderFactory(config)
    provider = factory.create_provider("openai")

    print(f"Created provider: {provider.get_provider_name()}")

    # Use the provider directly
    known_paths = ["/api/users", "/api/posts"]
    result = provider.generate_paths(known_paths, "blog.example.com", 5)

    print(f"Generated paths using {provider.get_provider_name()}:")
    for path, confidence in zip(result.paths, result.confidence_scores):
        print(f"  {path} (confidence: {confidence:.3f})")


async def main():
    """Run all examples."""
    print("Enumeraite Usage Examples")
    print("=" * 50)

    # Basic generation
    await basic_generation_example()

    # Validation example (requires internet connection)
    await validation_example()

    # Continuous discovery (requires internet connection)
    await continuous_discovery_example()

    # Manual provider creation
    manual_provider_example()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())