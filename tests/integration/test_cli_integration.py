"""Small end-to-end CLI tests with a deterministic fake provider."""

import json
from pathlib import Path

from click.testing import CliRunner

from enumeraite.cli.main import cli
from enumeraite.core.models import GenerationResult


class FakeProvider:
    model = "fake-model"

    def get_provider_name(self):
        return "fake"

    def generate_paths(self, known, target, count, max_depth=None, strategy=None):
        values = ["/api/v2/users", "/api/v2/orders"][:count]
        return GenerationResult(
            paths=values,
            confidence_scores=[0.9, 0.6][: len(values)],
            metadata={"provider": "fake", "model": self.model, "target": target},
        )


def _wire_fake(monkeypatch):
    monkeypatch.setattr("enumeraite.cli.run_command.detect_provider", lambda config: "fake")
    monkeypatch.setattr(
        "enumeraite.cli.run_command.ProviderFactory.create_provider",
        lambda self, name, overrides=None: FakeProvider(),
    )


def test_run_is_pipe_clean(tmp_path: Path, monkeypatch):
    _wire_fake(monkeypatch)
    seed = tmp_path / "paths.txt"
    seed.write_text("/api/v1/users\n/api/v1/orders\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["run", str(seed), "--target", "example.com", "-q"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["/api/v2/users", "/api/v2/orders"]


def test_run_json_output_file(tmp_path: Path, monkeypatch):
    _wire_fake(monkeypatch)
    seed = tmp_path / "paths.txt"
    output = tmp_path / "result.json"
    seed.write_text("/api/v1/users\n/api/v1/orders\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(seed),
            "--target",
            "example.com",
            "--json",
            "-o",
            str(output),
            "-q",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidates"][0] == {"value": "/api/v2/users", "confidence": 0.9}
    assert payload["metadata"]["provider"] == "fake"


def test_run_json_sanitizes_nonstandard_provider_metadata(tmp_path: Path, monkeypatch):
    class UnsafeProvider(FakeProvider):
        def generate_paths(self, known, target, count, max_depth=None, strategy=None):
            cycle = {}
            cycle["self"] = cycle
            return GenerationResult(
                paths=["/safe"],
                confidence_scores=[0.8],
                metadata={
                    "bad_float": float("nan"),
                    "custom": object(),
                    "cycle": cycle,
                },
            )

    monkeypatch.setattr("enumeraite.cli.run_command.detect_provider", lambda config: "fake")
    monkeypatch.setattr(
        "enumeraite.cli.run_command.ProviderFactory.create_provider",
        lambda self, name, overrides=None: UnsafeProvider(),
    )
    seed = tmp_path / "paths.txt"
    seed.write_text("/seed\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["run", str(seed), "--json", "-q"])
    assert result.exit_code == 0, result.output
    payload = json.loads(
        result.output, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
    )
    assert payload["metadata"]["bad_float"] == 0.0
    assert payload["metadata"]["cycle"]["self"] == "<cycle>"


def test_run_selects_huggingface_finetune_for_detected_kind(tmp_path: Path, monkeypatch):
    captured = {}
    monkeypatch.setattr("enumeraite.cli.run_command.detect_provider", lambda config: "huggingface")
    monkeypatch.setattr(
        "enumeraite.cli.run_command.ProviderFactory.create_provider",
        lambda self, name, overrides=None: captured.update(overrides or {}) or FakeProvider(),
    )
    seed = tmp_path / "paths.txt"
    seed.write_text("/api/v1/users\n/api/v1/orders\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["run", str(seed), "-q"])
    assert result.exit_code == 0, result.output
    assert captured["model"].endswith("-Path")


def test_run_rejects_credential_target_before_provider_creation(tmp_path: Path, monkeypatch):
    called = {"provider": False}
    monkeypatch.setattr("enumeraite.cli.run_command.detect_provider", lambda config: "fake")

    def create(*args, **kwargs):
        called["provider"] = True
        return FakeProvider()

    monkeypatch.setattr("enumeraite.cli.run_command.ProviderFactory.create_provider", create)
    seed = tmp_path / "paths.txt"
    seed.write_text("/api/users\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        ["run", str(seed), "--target", "https://user:secret@example.test", "-q"],
    )
    assert result.exit_code != 0
    assert "credentials" in result.output
    assert not called["provider"]


def test_benchmark_baseline_end_to_end(tmp_path: Path):
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "demo.json").write_text(
        json.dumps(
            {
                "openapi": "3.0.0",
                "info": {"title": "Demo", "version": "1"},
                "paths": {
                    "/api/v1/users": {},
                    "/api/v1/orders": {},
                    "/api/v2/users": {},
                    "/api/v2/orders": {},
                    "/api/v1/status": {},
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        ["bench", "openapi", "--specs", str(specs), "--count", "10", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.json").exists()
    assert (out / "report.md").exists()
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["engines"]["mutation"]["metadata"]["type"] == "baseline"
    assert set(payload["engines"]) == {"structural", "mutation-only", "mutation"}

    invalid = CliRunner().invoke(
        cli,
        [
            "bench",
            "openapi",
            "--specs",
            str(specs),
            "--budgets",
            "0",
            "-o",
            str(tmp_path / "invalid"),
        ],
    )
    assert invalid.exit_code != 0
    assert "at least one positive integer" in invalid.output


def test_legacy_generate_writes_output_file(tmp_path: Path, monkeypatch):
    seed = tmp_path / "paths.txt"
    output = tmp_path / "generated.txt"
    seed.write_text("/api/v1/users\n/api/v1/orders\n", encoding="utf-8")
    monkeypatch.setattr(
        "enumeraite.cli.commands._setup_provider",
        lambda ctx, provider, model, generation_type=None: (None, FakeProvider()),
    )
    result = CliRunner().invoke(
        cli, ["generate", "path", "-i", str(seed), "-o", str(output), "-c", "2"]
    )
    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8").splitlines() == [
        "/api/v2/users",
        "/api/v2/orders",
    ]


def test_legacy_generate_rejects_non_positive_budget(tmp_path: Path):
    seed = tmp_path / "paths.txt"
    seed.write_text("/api/v1/users\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["generate", "path", "-i", str(seed), "-c", "0"])
    assert result.exit_code != 0
    assert "0 is not in the range" in result.output


def test_legacy_subdomain_requires_unambiguous_target_and_validation_dependency(
    tmp_path: Path,
):
    single = tmp_path / "single.txt"
    single.write_text("api.example.co.uk\n", encoding="utf-8")
    ambiguous = CliRunner().invoke(cli, ["generate", "subdomain", "-i", str(single)])
    assert ambiguous.exit_code != 0
    assert "pass --target" in ambiguous.output

    siblings = tmp_path / "siblings.txt"
    siblings.write_text("api.example.com\ncdn.example.com\n", encoding="utf-8")
    dependency = CliRunner().invoke(
        cli,
        ["generate", "subdomain", "-i", str(siblings), "--check-http"],
    )
    assert dependency.exit_code != 0
    assert "requires --validate" in dependency.output


def test_legacy_generate_does_not_claim_failed_output_write(tmp_path: Path, monkeypatch):
    seed = tmp_path / "paths.txt"
    seed.write_text("/api/v1/users\n", encoding="utf-8")
    monkeypatch.setattr(
        "enumeraite.cli.commands._setup_provider",
        lambda ctx, provider, model, generation_type=None: (None, FakeProvider()),
    )
    result = CliRunner().invoke(
        cli,
        [
            "generate",
            "path",
            "-i",
            str(seed),
            "-o",
            str(tmp_path / "missing" / "output.txt"),
        ],
    )
    assert result.exit_code != 0
    assert "Generation failed" in result.output
    assert "Saved to" not in result.output
