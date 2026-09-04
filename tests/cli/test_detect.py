import pytest

from enumeraite.cli.detect import (
    _hf_usable,
    _local_endpoint,
    detect_kind,
    detect_provider,
    infer_apex,
)
from enumeraite.core.config import Config


def test_detect_kind_and_confidence():
    assert detect_kind(["/api/users", "/admin"])[0] == "path"
    kind, confidence = detect_kind(["api.example.com", "dev.example.com", "/mixed"])
    assert kind == "subdomain"
    assert confidence == 2 / 3


def test_detection_rejects_ambiguous_or_unbounded_direct_inputs():
    with pytest.raises(ValueError, match="iterable of strings"):
        detect_kind("/api")
    with pytest.raises(ValueError, match="only strings"):
        infer_apex(["api.example.com", 7])
    with pytest.raises(ValueError, match="100000-item"):
        detect_kind((f"/{index}" for index in range(100_001)))


def test_infer_apex_from_siblings():
    assert infer_apex(["api.dev.example.com", "cdn.example.com"]) == "example.com"


def test_infer_apex_refuses_single_or_mixed_scope_inputs():
    assert infer_apex(["api.example.co.uk"]) == ""
    assert infer_apex(["api.one.example", "cdn.two.example"]) == ""
    assert infer_apex(["https://api.example.com", "https://cdn.example.com"]) == ""


def test_unusable_huggingface_is_not_auto_selected(monkeypatch):
    monkeypatch.setattr("enumeraite.cli.detect._hf_usable", lambda: False)
    config = Config(providers={"huggingface": {"model": "large"}})
    assert detect_provider(config) is None


def test_huggingface_detection_enforces_declared_runtime_floors(monkeypatch):
    monkeypatch.setattr("enumeraite.cli.detect.importlib.util.find_spec", lambda package: object())
    versions = {
        "torch": "2.12.1+cpu",
        "transformers": "5.10.0",
        "accelerate": "0.26.0",
    }
    monkeypatch.setattr(
        "enumeraite.cli.detect.importlib.metadata.version", lambda package: versions[package]
    )
    assert not _hf_usable()
    versions["transformers"] = "5.10.1"
    assert not _hf_usable()
    versions["torch"] = "2.13.0"
    assert _hf_usable()


def test_unreachable_local_server_is_not_auto_selected(monkeypatch):
    monkeypatch.setattr("enumeraite.cli.detect.local_server_reachable", lambda *args: False)
    config = Config(providers={"ollama": {"model": "qwen"}})
    assert detect_provider(config) is None


def test_local_detection_honors_configured_base_url(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "enumeraite.cli.detect.local_server_reachable",
        lambda host, port: captured.append((host, port)) or True,
    )
    config = Config(
        default_provider="vllm",
        providers={
            "vllm": {
                "model": "local",
                "base_url": "http://model-box.example:9123/v1",
            }
        },
    )
    assert detect_provider(config) == "vllm"
    assert captured == [("model-box.example", 9123)]
    assert _local_endpoint("https://localhost/v1") == ("localhost", 443)
    assert _local_endpoint("http://user@localhost/v1") is None


def test_explicit_default_provider_wins_when_usable():
    config = Config(
        default_provider="minimax",
        providers={
            "openai": {"api_key": "openai-test"},
            "minimax": {"api_key": "minimax-test"},
        },
    )
    assert detect_provider(config) == "minimax"


def test_hosted_provider_without_key_is_not_usable():
    config = Config(providers={"openai": {"model": "gpt-4o-mini"}})
    assert detect_provider(config) is None
