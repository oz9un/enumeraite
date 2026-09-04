#!/usr/bin/env python3
"""Minimal programmatic Enumeraite example."""

from enumeraite.cli.detect import detect_provider
from enumeraite.core.config import load_config
from enumeraite.core.engine import GenerationEngine
from enumeraite.core.factory import ProviderFactory
from enumeraite.core.strategies import PathDiscoveryStrategy


def main() -> None:
    config = load_config()
    provider_name = detect_provider(config)
    if not provider_name:
        raise SystemExit("No usable provider. Run `enumeraite doctor` for setup guidance.")

    provider = ProviderFactory(config).create_provider(provider_name)
    engine = GenerationEngine(provider)
    known = ["/api/v1/users", "/api/v1/orders", "/api/v1/auth/login"]
    result = engine.generate_paths(
        known,
        target="api.example.com",
        count=10,
        strategy=PathDiscoveryStrategy(),
    )

    print(f"Provider: {provider.get_provider_name()}")
    for path, confidence in zip(result.paths, result.confidence_scores):
        print(f"{confidence:.2f}\t{path}")


if __name__ == "__main__":
    main()
