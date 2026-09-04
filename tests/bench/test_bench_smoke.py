import json

import pytest

from enumeraite.bench.baselines import (
    mutation_baseline,
    mutation_only_baseline,
    structural_baseline,
    wordlist_baseline,
)
from enumeraite.bench.dataset import (
    BenchCase,
    _seeded_split,
    load_openapi_dataset,
    load_subdomain_dataset,
)
from enumeraite.bench.metrics import _mean_ci95, _path_matches_template, aggregate, score_run
from enumeraite.bench.runner import (
    BenchRunner,
    RunConfig,
    make_baseline_engine,
    make_llm_engine,
    replay_benchmark,
)
from enumeraite.core.models import GenerationResult
from enumeraite.core.strategies import PathDiscoveryStrategy


def test_seed_split_is_deterministic_and_never_leaks():
    items = [f"/item/{index}" for index in range(10)]
    first = _seeded_split(items, 0.2, "salt")
    second = _seeded_split(reversed(items), 0.2, "salt")
    assert first == second
    assert not set(first[0]) & set(first[1])
    with pytest.raises(ValueError, match="leaked"):
        BenchCase("x", "path", ["/same"], ["/same"], ["/same"])


@pytest.mark.parametrize(
    "call",
    [
        lambda: _seeded_split("/one", 0.5, "salt"),
        lambda: _seeded_split(["/one", "/two"], True, "salt"),
        lambda: _seeded_split(["/one", "/two"], 0.5, ""),
        lambda: _seeded_split(["/one", "/two"], 0.5, "salt", max_seed=True),
        lambda: BenchCase("x", "path", "/seed", ["/hit"], ["/seed", "/hit"]),
    ],
)
def test_dataset_direct_api_rejects_ambiguous_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_route_shape_aliases_cannot_leak_across_split():
    with pytest.raises(ValueError, match="route shape leaked"):
        BenchCase(
            "x",
            "path",
            ["/users/{id}"],
            ["/users/{userId}"],
            ["/users/{id}", "/users/{userId}"],
        )


def test_openapi_loader_deduplicates_equivalent_route_templates(tmp_path):
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Aliases", "version": "1"},
        "paths": {
            "/users/{id}": {},
            "/users/{userId}": {},
            "/orders": {},
            "/products": {},
            "/status": {},
        },
    }
    (tmp_path / "aliases.json").write_text(json.dumps(spec), encoding="utf-8")
    cases = load_openapi_dataset(tmp_path, seed_frac=0.5)
    assert len(cases) == 1
    assert len(cases[0].universe) == 4
    assert not ({"/users/{id}", "/users/{userId}"} <= set(cases[0].universe))


def test_small_sample_interval_uses_student_t_uncertainty():
    lower, upper = _mean_ci95([0.4, 0.5, 0.5, 0.5, 0.6])
    assert lower == pytest.approx(0.4122, abs=0.0002)
    assert upper == pytest.approx(0.5878, abs=0.0002)


def test_repeated_splits_are_clustered_for_confidence_interval():
    metrics = []
    for source in ("source-a", "source-b"):
        case = BenchCase(
            source,
            "path",
            ["/seed"],
            ["/hit"],
            ["/seed", "/hit"],
            meta={"source": source},
        )
        metrics.append(score_run(case, ["/hit"], [0.9], budgets=(1,)))
        metrics.append(score_run(case, ["/miss"], [0.9], budgets=(1,)))
    result = aggregate(metrics, (1,))
    assert result["macro_recall_at_budget"][1] == 0.5
    assert result["macro_recall_ci95"][1] == [0.5, 0.5]
    assert result["ci_clusters"] == 2


def test_identical_source_hashes_share_one_confidence_cluster():
    metrics = []
    for source in ("copy-a.json", "copy-b.json"):
        case = BenchCase(
            source,
            "path",
            ["/seed"],
            ["/hit"],
            ["/seed", "/hit"],
            meta={"source": source, "source_sha256": "same-content-hash"},
        )
        metrics.append(score_run(case, ["/hit"], [0.9], budgets=(1,)))
    result = aggregate(metrics, (1,))
    assert result["ci_clusters"] == 1


def test_empty_successful_case_does_not_improve_brier_calibration():
    case = BenchCase("demo", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])
    emitted = score_run(case, ["/hit"], [0.9], budgets=(1,))
    empty = score_run(case, [], [], budgets=(1,))
    result = aggregate([emitted, empty], (1,))
    assert result["calibrated_cases"] == 1
    assert result["mean_brier_score"] == pytest.approx(0.01)


def test_non_finite_confidence_cannot_corrupt_ranking_or_json_trace():
    case = BenchCase("demo", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])
    metrics = score_run(case, ["/hit", "/miss"], [float("nan"), 0.5], budgets=(1, 2))
    assert metrics.recall_at_budget == {1: 0.0, 2: 1.0}
    assert metrics.ranked_candidates[1]["confidence"] == 0.0


def test_engine_output_is_hard_capped_to_the_requested_budget():
    case = BenchCase(
        "demo",
        "path",
        ["/seed"],
        ["/one", "/two", "/three"],
        ["/seed", "/one", "/two", "/three"],
    )
    metrics = score_run(
        case,
        ["/one", "/two", "/three"],
        [0.9, 0.8, 0.7],
        budgets=(2,),
        requested=2,
    )
    assert metrics.generated == 2
    assert metrics.valid == 2
    assert metrics.fill_rate == 1.0


def test_candidates_beyond_raw_budget_cannot_compete_in_ranking():
    case = BenchCase("demo", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])
    metrics = score_run(
        case,
        ["/miss", "/hit"],
        [0.1, 1.0],
        budgets=(1,),
        requested=1,
    )
    assert metrics.generated == 1
    assert metrics.valid == 0
    assert metrics.ranked_candidates[0]["value"] == "/miss"


def test_hybrid_hits_are_attributed_and_deduplicated_by_source():
    case = BenchCase(
        "demo",
        "path",
        ["/seed"],
        ["/pets/{petId}", "/other"],
        ["/seed", "/pets/{petId}", "/other"],
    )
    metrics = score_run(
        case,
        ["/pets/1", "/pets/2", "/other"],
        [0.9, 0.8, 0.7],
        budgets=(3,),
        generation_meta={
            "candidate_sources": {
                "/pets/1": "structural",
                "/pets/2": "model",
                "/other": "model",
            }
        },
    )
    assert metrics.valid_by_source == {"model": 2, "structural": 1}
    assert metrics.unique_valid_by_source == {"model": 1, "structural": 0}


def test_template_metrics_rank_by_confidence():
    case = BenchCase(
        "demo",
        "path",
        ["/pets"],
        ["/pets/{petId}", "/users"],
        ["/pets", "/pets/{petId}", "/users"],
    )
    assert _path_matches_template("/pets/123", "/pets/{petId}")
    assert not _path_matches_template("/pets/create", "/pets/{petId}")
    assert _path_matches_template("/", "/")
    metrics = score_run(case, ["/users", "/pets/123"], [0.1, 0.9], budgets=(1, 2))
    assert metrics.recall_at_budget == {1: 0.5, 2: 1.0}
    assert metrics.precision == 1.0


def test_metrics_do_not_reward_seed_or_template_repetitions():
    case = BenchCase(
        "demo",
        "path",
        ["/known"],
        ["/pets/{id}"],
        ["/known", "/pets/{id}"],
    )
    metrics = score_run(
        case,
        ["/known", "/pets/1", "/pets/2", "/miss"],
        [1.0, 0.9, 0.8, 0.1],
        budgets=(1, 3),
        requested=4,
    )
    assert metrics.generated == 3
    assert metrics.valid == 1
    assert metrics.precision == pytest.approx(1 / 3)
    assert metrics.recall_at_budget == {1: 1.0, 3: 1.0}
    assert metrics.fill_rate == 0.75


def test_mutation_baseline_instantiates_openapi_templates():
    candidates = mutation_baseline(
        ["/api/v1/users/{userId}", "/api/v1/orders/:orderId/status"],
        50,
        "path",
    )
    assert candidates
    assert all("{" not in candidate and ":orderId" not in candidate for candidate in candidates)
    assert "/api/v1/users/123/settings" in candidates


def test_mutation_baseline_never_spends_budget_repeating_visible_seeds():
    subdomains = mutation_baseline(
        ["api-dev.example.com", "api-prod.example.com"],
        50,
        "subdomain",
        "example.com",
    )
    assert "api-dev.example.com" not in subdomains
    assert "api-prod.example.com" not in subdomains

    paths = mutation_baseline(["/users/{userId}"], 50, "path")
    assert "/users/123" not in paths


def test_structural_and_mutation_only_controls_are_true_ablations():
    known = ["/api/v1/users/profile", "/api/v2/orders"]
    structural = structural_baseline(known, 50, "path")
    mutation_only = mutation_only_baseline(known, 50, "path")
    combined = mutation_baseline(known, 50, "path")

    assert "/api/v1/orders" in structural
    assert "/api/v1/users" not in mutation_only
    assert "/api/v1/users/profile/settings" in mutation_only
    assert combined[: len(structural)] == structural
    first_lexical_only = next(value for value in mutation_only if value not in structural)
    assert first_lexical_only in combined


def test_baselines_reject_ambiguous_inputs_and_honor_zero_budget():
    assert mutation_baseline(["/seed"], 0, "path") == []
    assert wordlist_baseline(["admin"], kind="path", count=0) == []
    with pytest.raises(ValueError, match="not a string"):
        mutation_baseline("/seed", 10, "path")
    with pytest.raises(ValueError, match="between 0 and 5000"):
        structural_baseline(["/seed"], 5001, "path")
    with pytest.raises(ValueError, match="not a string"):
        wordlist_baseline("admin", kind="path")
    with pytest.raises(ValueError, match="valid apex"):
        wordlist_baseline(["admin"], apex="https://example.com", kind="subdomain")


def test_exact_literal_route_takes_precedence_over_template_match():
    case = BenchCase(
        "demo",
        "path",
        ["/seed"],
        ["/user/login", "/user/{username}"],
        ["/seed", "/user/login", "/user/{username}"],
    )
    metrics = score_run(case, ["/user/login", "/user/alice"], [0.9, 0.8], budgets=(1, 2))
    assert metrics.recall_at_budget == {1: 0.5, 2: 1.0}
    assert metrics.valid == 2


def test_general_template_rejects_reserved_words_case_insensitively():
    case = BenchCase(
        "demo",
        "path",
        ["/seed"],
        ["/users/{username}"],
        ["/seed", "/users/{username}"],
    )
    metrics = score_run(case, ["/users/Login", "/users/alice"], [0.9, 0.8], budgets=(2,))
    assert metrics.valid == 1
    assert metrics.ranked_candidates[0]["matches"] == []
    assert metrics.ranked_candidates[1]["matches"] == ["/users/{username}"]


def test_literal_openapi_placeholder_is_not_credited_as_a_concrete_candidate():
    case = BenchCase(
        "api.test",
        "path",
        ["/pets"],
        ["/pets/{petId}"],
        ["/pets", "/pets/{petId}"],
    )

    metrics = score_run(case, ["/pets/{petId}"], [1.0], budgets=(1,))

    assert metrics.generated == 1
    assert metrics.valid == 0
    assert metrics.precision == 0.0


def test_one_candidate_cannot_cover_two_overlapping_templates():
    case = BenchCase(
        "demo",
        "path",
        ["/seed"],
        ["/orgs/{org}/requests/{requestId}", "/orgs/{org}/{product}/{action}"],
        ["/seed", "/orgs/{org}/requests/{requestId}", "/orgs/{org}/{product}/{action}"],
    )
    metrics = score_run(
        case,
        ["/orgs/acme/requests/123"],
        [0.9],
        budgets=(1,),
    )
    assert metrics.valid == 1
    assert metrics.recall_at_budget == {1: 0.5}
    assert metrics.novel_valid_items == ["/orgs/{org}/requests/{requestId}"]


def test_path_benchmark_preserves_case_but_dns_does_not():
    path_case = BenchCase("x", "path", ["/API"], ["/api"], ["/API", "/api"])
    assert score_run(path_case, ["/api"], [1.0], budgets=(1,)).valid == 1
    with pytest.raises(ValueError, match="leaked"):
        BenchCase(
            "example.com",
            "subdomain",
            ["API.Example.com"],
            ["api.example.com"],
            ["api.example.com"],
        )


def test_path_wordlist_uniqueness_preserves_route_case():
    case = BenchCase("x", "path", ["/seed"], ["/api"], ["/seed", "/api"])
    metrics = score_run(
        case,
        ["/api"],
        [1.0],
        budgets=(1,),
        baseline_wordlist=["/API"],
    )
    assert metrics.valid == 1
    assert metrics.wordlist_unique_items == ["/api"]


def test_openapi_loader_and_runner_write_reports(tmp_path):
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Demo", "version": "1"},
        "paths": {"/api/v1/a": {}, "/api/v1/b": {}, "/api/v2/a": {}, "/api/v2/b": {}},
    }
    (tmp_path / "demo.json").write_text(json.dumps(spec), encoding="utf-8")
    cases = load_openapi_dataset(tmp_path)
    assert len(cases) == 1
    out = tmp_path / "results"
    engine = make_baseline_engine(
        lambda known, count, target: mutation_baseline(known, count, "path", target)
    )
    result = BenchRunner(
        RunConfig(
            count=20,
            budgets=(5, 20),
            out_dir=out,
            engine_metadata={"mutation": {"model": "deterministic-v1"}},
        )
    ).run(cases, {"mutation": engine})
    assert result["schema_version"] == 2
    assert result["metric_version"] == "route-shape-v10"
    assert result["runtime"]["git_dirty"] in {True, False, None}
    assert result["config"]["pricing_checked_at"] == "2026-09-02"
    assert result["engines"]["mutation"]["metadata"]["model"] == "deterministic-v1"
    assert result["engines"]["mutation"]["aggregate"]["cases"] == 1
    assert result["dataset"][0]["known"] == cases[0].known
    assert result["dataset"][0]["hidden"] == cases[0].hidden
    assert (out / "results.json").exists()
    assert "Recall@20" in (out / "report.md").read_text(encoding="utf-8")
    assert "deterministic-v1" in (out / "report.md").read_text(encoding="utf-8")

    replayed = replay_benchmark(out / "results.json", tmp_path / "replayed")
    assert replayed["metric_version"] == "route-shape-v10"
    assert replayed["config"]["source_metric_version"] == "route-shape-v10"
    assert len(replayed["config"]["source_results_sha256"]) == 64
    assert (
        replayed["engines"]["mutation"]["aggregate"]["valid"]
        == result["engines"]["mutation"]["aggregate"]["valid"]
    )


def test_replay_can_use_the_exact_bytes_already_inspected(tmp_path):
    case = BenchCase("demo", "path", ["/known"], ["/hidden"], ["/known", "/hidden"])
    out = tmp_path / "original"
    BenchRunner(RunConfig(count=1, budgets=(1,), out_dir=out)).run(
        [case], {"mutation": lambda *_args: (["/hidden"], [1.0], {})}
    )
    results = out / "results.json"
    inspected = results.read_bytes()
    results.write_text("not the inspected result", encoding="utf-8")

    replayed = replay_benchmark(
        results,
        tmp_path / "replayed",
        source_bytes=inspected,
    )

    assert replayed["engines"]["mutation"]["aggregate"]["valid"] == 1


def test_failed_cases_count_as_zero_macro_coverage(tmp_path):
    cases = [
        BenchCase("one", "path", ["/seed"], ["/hit"], ["/seed", "/hit"]),
        BenchCase("two", "path", ["/seed"], ["/hit"], ["/seed", "/hit"]),
    ]
    calls = {"count": 0}

    def unreliable(known, count, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise TimeoutError("provider timeout")
        return ["/hit"], [0.9], {}

    result = BenchRunner(RunConfig(count=1, budgets=(1,), out_dir=tmp_path / "result")).run(
        cases, {"unreliable": unreliable}
    )
    aggregate_result = result["engines"]["unreliable"]["aggregate"]
    assert aggregate_result["cases"] == 2
    assert aggregate_result["successful_cases"] == 1
    assert aggregate_result["errors"] == 1
    assert aggregate_result["macro_recall_at_budget"][1] == 0.5
    assert aggregate_result["macro_average_precision"] == 0.5
    assert aggregate_result["mean_fill_rate"] == 0.5
    assert aggregate_result["cost"]["seconds"] >= 0
    assert result["engines"]["unreliable"]["cases"][1]["cost_per_valid"] == {"seconds": None}

    replayed = replay_benchmark(tmp_path / "result" / "results.json", tmp_path / "failed-replay")
    original_case = result["engines"]["unreliable"]["cases"][1]
    replayed_case = replayed["engines"]["unreliable"]["cases"][1]
    assert replayed_case["error"] == original_case["error"]
    assert replayed_case["cost"] == original_case["cost"]


def test_provider_failure_artifacts_redact_api_key_shaped_tokens(tmp_path):
    case = BenchCase("one", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])

    def failed(known, count, target):
        raise RuntimeError("request failed for sk-api-abcdefghijklmnopqrstuvwxyz123456")

    result = BenchRunner(RunConfig(count=1, budgets=(1,), out_dir=tmp_path)).run(
        [case], {"failed": failed}
    )
    error = result["engines"]["failed"]["cases"][0]["error"]
    assert "sk-api-" not in error
    assert "<redacted-api-key>" in error


@pytest.mark.parametrize(
    "message",
    [
        "request failed with gsk_examplecredential123456",
        "download failed for hf_examplecredential123456",
        "headers={'api_key': 'plain-credential-123456'}",
        "Authorization=opaque-credential-123456",
    ],
)
def test_provider_failure_artifacts_redact_additional_credential_shapes(tmp_path, message):
    case = BenchCase("one", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])

    def fail(*_args):
        raise RuntimeError(message)

    result = BenchRunner(RunConfig(count=1, budgets=(1,), out_dir=tmp_path)).run(
        [case], {"provider": fail}
    )
    error = result["engines"]["provider"]["cases"][0]["error"]

    assert "credential" not in error
    assert "<redacted" in error


def test_replay_refuses_misaligned_or_malformed_traces(tmp_path):
    case = BenchCase("one", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])
    out = tmp_path / "source"
    BenchRunner(RunConfig(count=1, budgets=(1,), out_dir=out)).run(
        [case],
        {"engine": make_baseline_engine(lambda known, count, target: ["/hit"])},
    )
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    payload["engines"]["engine"]["cases"][0]["kind"] = "subdomain"
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ordering differs"):
        replay_benchmark(broken, tmp_path / "replay")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("count", True, "candidate count"),
        ("count", 1.5, "candidate count"),
        ("budgets", [True], "candidate budget"),
        ("budgets", [1.5], "candidate budget"),
    ],
)
def test_replay_rejects_coerced_budget_values(tmp_path, field, value, message):
    case = BenchCase("one", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])
    out = tmp_path / "source"
    BenchRunner(RunConfig(count=1, budgets=(1,), out_dir=out)).run(
        [case],
        {"engine": make_baseline_engine(lambda known, count, target: ["/hit"])},
    )
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    payload["config"][field] = value
    broken = tmp_path / f"broken-{field}.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        replay_benchmark(broken, tmp_path / "replay")


def test_core_replay_requires_the_recorded_wordlist_hash(tmp_path):
    case = BenchCase("one", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])
    out = tmp_path / "source"
    expected_hash = "a" * 64
    BenchRunner(
        RunConfig(
            count=1,
            budgets=(1,),
            out_dir=out,
            experiment={"wordlist_sha256": expected_hash},
        )
    ).run([case], {"engine": make_baseline_engine(lambda known, count, target: ["/hit"])})
    with pytest.raises(ValueError, match="requires its original wordlist"):
        replay_benchmark(out / "results.json", tmp_path / "missing")
    with pytest.raises(ValueError, match="does not match"):
        replay_benchmark(
            out / "results.json",
            tmp_path / "wrong",
            baseline_wordlist=["hit"],
            wordlist_sha256="b" * 64,
        )
    replayed = replay_benchmark(
        out / "results.json",
        tmp_path / "correct",
        baseline_wordlist=["hit"],
        wordlist_sha256=expected_hash,
    )
    assert replayed["config"]["wordlist_sha256"] == expected_hash


@pytest.mark.parametrize(
    "config",
    [
        RunConfig(count=0),
        RunConfig(count=True),
        RunConfig(count=2.5),
        RunConfig(budgets=()),
        RunConfig(budgets=(True,)),
        RunConfig(budgets=(1.5,)),
        RunConfig(budgets=(5_001,)),
        RunConfig(budgets=(1, 1)),
    ],
)
def test_runner_rejects_invalid_direct_api_budgets(config):
    with pytest.raises(ValueError, match="benchmark"):
        BenchRunner(config)


@pytest.mark.parametrize(
    "config",
    [
        RunConfig(pricing={"model": (-1.0, 1.0)}),
        RunConfig(pricing={"model": (float("nan"), 1.0)}),
        RunConfig(pricing={"model": (1.0,)}),
        RunConfig(cache_read_pricing={"model": float("inf")}),
    ],
)
def test_runner_rejects_invalid_pricing_metadata(config):
    with pytest.raises(ValueError, match="pric"):
        BenchRunner(config)


def test_experiment_metadata_cannot_override_manifest_contract(tmp_path):
    case = BenchCase("demo", "path", ["/seed"], ["/hit"], ["/seed", "/hit"])
    result = BenchRunner(
        RunConfig(
            count=1,
            budgets=(1,),
            out_dir=tmp_path,
            experiment={
                "count": 999,
                "budgets": [999],
                "pricing_source": "untrusted",
                "custom_note": "preserved",
            },
        )
    ).run([case], {"engine": make_baseline_engine(lambda known, count, target: ["/hit"])})
    assert result["config"]["count"] == 1
    assert result["config"]["budgets"] == [1]
    assert result["config"]["pricing_source"] != "untrusted"
    assert result["config"]["custom_note"] == "preserved"


def test_runner_reports_clustered_paired_delta_against_mutation(tmp_path):
    cases = [
        BenchCase(
            name,
            "path",
            ["/seed"],
            ["/one", "/two"],
            ["/seed", "/one", "/two"],
            meta={"source": name},
        )
        for name in ("source-a", "source-b")
    ]
    mutation = make_baseline_engine(lambda known, count, target: ["/one"])
    improved = make_baseline_engine(lambda known, count, target: ["/one", "/two"])
    result = BenchRunner(RunConfig(count=2, budgets=(2,), out_dir=tmp_path / "paired")).run(
        cases, {"mutation": mutation, "improved": improved}
    )
    comparison = result["engines"]["improved"]["aggregate"]["comparison_to_mutation"]
    assert comparison["recall_delta"] == 0.5
    assert comparison["recall_delta_ci95"] == [0.5, 0.5]
    assert comparison["paired_clusters"] == 2
    assert "Δ recall vs mutation" in (tmp_path / "paired" / "report.md").read_text(encoding="utf-8")


def test_llm_cost_applies_known_cache_read_rate():
    class Provider:
        def generate_paths(self, known, target, count, strategy=None):
            return GenerationResult(
                paths=["/hit"],
                confidence_scores=[0.9],
                metadata={
                    "model": "MiniMax-M3",
                    "candidate_sources": {"/hit": "model"},
                    "token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "output_tokens": 10,
                    },
                },
            )

    _, _, cost, generation_meta = make_llm_engine(Provider(), PathDiscoveryStrategy())(
        ["/seed"], 1, "example.test"
    )
    assert cost["new_input_tokens"] == 20
    assert cost["cached_input_tokens"] == 80
    assert cost["model_candidates"] == 1
    assert cost["structural_candidates"] == 0
    assert cost["usd"] == pytest.approx(22.8 / 1_000_000)
    assert generation_meta["model"] == "MiniMax-M3"
    assert generation_meta["candidate_sources"] == {"/hit": "model"}


def test_llm_engine_sanitizes_malformed_usage_and_rejects_bad_custom_prices():
    class Provider:
        def generate_paths(self, known, target, count, strategy=None):
            return GenerationResult(
                paths=["/hit"],
                confidence_scores=[0.9],
                metadata={"model": "custom", "token_usage": "not-an-object"},
            )

    _, _, cost, _ = make_llm_engine(Provider(), PathDiscoveryStrategy())(
        ["/seed"], 1, "example.test"
    )
    assert cost["input_tokens"] == 0
    assert cost["output_tokens"] == 0
    with pytest.raises(ValueError, match="finite and non-negative"):
        make_llm_engine(
            Provider(),
            PathDiscoveryStrategy(),
            pricing={"custom": (-1.0, 1.0)},
        )


def test_openapi_loader_accepts_yaml(tmp_path):
    (tmp_path / "demo.yaml").write_text(
        """openapi: 3.0.0
info:
  title: YAML Demo
  version: '1'
paths:
  /one: {}
  /two: {}
  /three: {}
  /four: {}
""",
        encoding="utf-8",
    )
    cases = load_openapi_dataset(tmp_path)
    assert len(cases) == 1
    assert cases[0].target == "YAML Demo"


def test_openapi_loader_excludes_unsafe_ground_truth_paths(tmp_path):
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Scoped", "version": "1"},
        "paths": {
            "/one": {},
            "/two": {},
            "/three": {},
            "/four": {},
            "/safe?debug=true": {},
            "/../../etc/passwd": {},
        },
    }
    (tmp_path / "scoped.json").write_text(json.dumps(spec), encoding="utf-8")

    cases = load_openapi_dataset(tmp_path)

    assert set(cases[0].universe) == {"/one", "/two", "/three", "/four"}
    assert "/safe?debug=true" not in cases[0].universe
    assert "/../../etc/passwd" not in cases[0].universe


def test_swagger_v2_loader_uses_host_and_base_path(tmp_path):
    spec = {
        "swagger": "2.0",
        "info": {"title": "Legacy", "version": "1"},
        "schemes": ["http"],
        "host": "api.example.test",
        "basePath": "/v1",
        "paths": {"/one": {}, "/two": {}, "/three": {}, "/four": {}},
    }
    (tmp_path / "swagger.json").write_text(json.dumps(spec), encoding="utf-8")
    cases = load_openapi_dataset(tmp_path)
    assert cases[0].target == "http://api.example.test/v1"
    assert cases[0].meta["spec_version"] == "2.0"


def test_openapi_loader_does_not_put_server_credentials_or_query_tokens_in_target(tmp_path):
    def write(name, server):
        payload = {
            "openapi": "3.0.0",
            "info": {"title": name, "version": "1"},
            "servers": [{"url": server}],
            "paths": {"/one": {}, "/two": {}, "/three": {}, "/four": {}},
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    write("credentialed", "https://user:secret@api.example.test/v1")
    write("query", "https://api.example.test/v1?token=secret#fragment")
    cases = {case.meta["title"]: case for case in load_openapi_dataset(tmp_path)}
    assert cases["credentialed"].target == "credentialed"
    assert cases["query"].target == "https://api.example.test/v1"


def test_subdomain_loader_records_source_hash_and_split(tmp_path):
    (tmp_path / "acme.test.txt").write_text(
        "\n".join(f"host-{index}.acme.test" for index in range(6)),
        encoding="utf-8",
    )
    cases = load_subdomain_dataset(tmp_path, seed_frac=0.5, max_seed=2)
    assert len(cases) == 1
    assert cases[0].target == "acme.test"
    assert len(cases[0].known) == 2
    assert len(cases[0].meta["source_sha256"]) == 64
    assert "acme.test.txt" in cases[0].meta["split_salt"]


def test_subdomain_loader_excludes_malformed_and_cross_apex_truth(tmp_path):
    (tmp_path / "acme.test.txt").write_text(
        "API.ACME.TEST.\nweb.acme.test\noutside.example.test\nbad..acme.test\n",
        encoding="utf-8",
    )

    cases = load_subdomain_dataset(tmp_path)

    assert cases[0].universe == ["api.acme.test", "web.acme.test"]
