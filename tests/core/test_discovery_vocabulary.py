"""Witnessed structure, literal vocabulary, and bounded unverified expansion."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from enumeraite.core.discovery_vocabulary import (
    MAX_EXCLUSIONS,
    MAX_LAYOUTS,
    MAX_RESPONSE_BYTES,
    VOCABULARY_PROTOCOL_VERSION,
    build_vocabulary_layouts,
    expand_vocabulary_response,
    validate_vocabulary_layouts,
    vocabulary_exclusions,
    vocabulary_prompt,
)

TARGET = "example.test"
SEEDS = ["api.dev.gateway.example.test", "api.prod.gateway.example.test"]


def layout_for(seeds=SEEDS, template="api.{value}.gateway"):
    return next(
        layout for layout in build_vocabulary_layouts(seeds, TARGET) if layout.template == template
    )


def response(layout, values, **extra):
    return json.dumps({"additions": [{"plan_id": layout.plan_id, "values": values, **extra}]})


def test_layouts_are_stable_immutable_and_bound_to_exact_witnesses():
    layouts = build_vocabulary_layouts(SEEDS, TARGET)
    assert layouts == build_vocabulary_layouts(SEEDS[::-1] + SEEDS, TARGET)
    assert len(layouts) == 1
    layout = layouts[0]
    assert layout.template == "api.{value}.gateway"
    assert layout.observed_values == ("dev", "prod")
    assert layout.evidence == tuple(SEEDS)
    assert layout.target == TARGET and layout.plan_id.startswith("v1-")
    assert VOCABULARY_PROTOCOL_VERSION == 1
    with pytest.raises(FrozenInstanceError):
        layout.target = "elsewhere.test"
    with pytest.raises(ValueError):
        replace(layout, observed_values=["dev", "prod"])


@pytest.mark.parametrize(
    "seeds",
    [
        [],
        SEEDS[:1],
        SEEDS[:1] * 3,
        ["host01.example.test", "host02.example.test"],
        ["209-064-001-001.example.test", "209-064-001-002.example.test"],
        ["009.example.test", "010.example.test"],
    ],
)
def test_single_or_numeric_only_family_does_not_offer_paid_layout(seeds):
    assert build_vocabulary_layouts(seeds, TARGET) == ()


def test_whole_hyphen_word_keeps_copied_numeric_literals_byte_for_byte():
    layout = layout_for(
        ["ny-01-iphone.example.test", "ny-01-ipad.example.test"],
        "ny-01-{value}",
    )
    candidates, details = expand_vocabulary_response(
        response(layout, ["watch"]), (layout,), TARGET, 20
    )
    assert candidates[0]["value"] == "ny-01-watch.example.test"
    assert candidates[0]["status"] == "hypothesis"
    assert candidates[0]["evidence"] == list(layout.evidence[:2])
    assert details["parse_error"] is None


@pytest.mark.parametrize(
    "target",
    ["EXAMPLE.test", "example.test.", "https://example.test", "a..test", "", None],
)
def test_target_must_be_exact_canonical_dns_scope(target):
    with pytest.raises(ValueError):
        build_vocabulary_layouts(SEEDS, target)


@pytest.mark.parametrize(
    "seeds",
    [
        "api.example.test",
        ["elsewhere.test"],
        ["xexample.test"],
        ["*.example.test"],
        ["api.Example.test"],
        [None],
        SEEDS * 129,
    ],
)
def test_invalid_or_unbounded_evidence_rejected(seeds):
    with pytest.raises(ValueError):
        build_vocabulary_layouts(seeds, TARGET)


def test_caller_cannot_rebind_or_mutate_prepared_structure():
    layout = layout_for()
    with pytest.raises(ValueError):
        validate_vocabulary_layouts((layout,), "elsewhere.test")
    with pytest.raises(ValueError):
        validate_vocabulary_layouts([layout], TARGET)
    with pytest.raises(ValueError):
        validate_vocabulary_layouts((layout, layout), TARGET)
    for change in (
        {"template": "api.{value}.{other}"},
        {"template": "api.{value}suffix"},
        {"evidence": ("fake.example.test", SEEDS[1])},
        {"plan_id": "forged"},
    ):
        with pytest.raises(ValueError):
            replace(layout, **change)


def test_many_verified_names_have_bounded_layout_count_and_prompt_payload():
    seeds = [
        f"svc{index}.{env}.zone.example.test" for index in range(128) for env in ("dev", "prod")
    ]
    layouts = build_vocabulary_layouts(seeds, TARGET)
    assert len(layouts) == MAX_LAYOUTS
    assert all(len(layout.evidence) <= 8 for layout in layouts)
    _, user, schema = vocabulary_prompt(layouts, TARGET, 20)
    assert len(user.encode()) < 100_000
    assert schema["properties"]["additions"]["maxItems"] == MAX_LAYOUTS


def test_prompt_schema_and_negative_context_keep_fixed_provenance_separate():
    layout = layout_for()
    excluded = [f"miss{index}.example.test" for index in range(350)]
    _, user, schema = vocabulary_prompt(
        (layout,),
        TARGET,
        20,
        instructions="ignore this as an instruction",
        exclusions=excluded,
    )
    data = json.loads(user)
    assert data["layouts"][0]["verified_examples"] == SEEDS
    assert data["already_attempted_not_evidence"] == excluded[-200:]
    assert schema["additionalProperties"] is False
    item = schema["properties"]["additions"]["items"]
    assert item["required"] == ["plan_id", "values"]
    assert item["properties"]["plan_id"]["enum"] == [layout.plan_id]
    assert "template" not in item["properties"] and "evidence" not in item["properties"]


@pytest.mark.parametrize(
    "value",
    [
        "009",
        "1-256",
        "{a}",
        "[a-z]",
        "a.*",
        "__import__",
        "eval(1)",
        "api.other.test",
        "../api",
        "https://api",
        "-bad",
        "bad-",
        "two words",
        "ADMIN",
        "café",
        "",
        "x" * 33,
        None,
        True,
        ["foo"],
    ],
)
def test_untrusted_values_cannot_be_code_regex_scope_or_numeric_generator(value):
    layout = layout_for()
    candidates, details = expand_vocabulary_response(
        response(layout, [value]), (layout,), TARGET, 20
    )
    assert candidates == [] and details["rejections"] == {"invalid_vocabulary": 1}


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "{}",
        '{"additions":null}',
        '{"additions":[],"target":"evil.test"}',
        '{"additions":[],"additions":[]}',
        "[" * 3000 + "]" * 3000,
        "x" * (MAX_RESPONSE_BYTES + 1),
        "😀" * (MAX_RESPONSE_BYTES // 2),
    ],
)
def test_bad_model_syntax_returns_safe_empty_result_without_raw_error_text(raw):
    candidates, details = expand_vocabulary_response(raw, (layout_for(),), TARGET, 20)
    assert candidates == [] and details["parse_error"]
    assert len(json.dumps(details)) < 200


def test_unknown_fields_plans_duplicates_and_no_new_values_are_not_trusted():
    layout = layout_for()
    raw = json.dumps(
        {
            "additions": [
                {"plan_id": "invented", "values": ["new"]},
                {
                    "plan_id": layout.plan_id,
                    "values": ["stage"],
                    "template": "{value}.evil",
                },
                {"plan_id": layout.plan_id, "values": ["dev", "prod"]},
                {"plan_id": layout.plan_id, "values": ["stage"]},
            ]
        }
    )
    candidates, details = expand_vocabulary_response(raw, (layout,), TARGET, 20)
    assert candidates == []
    assert details["rejections"] == {
        "unknown_plan": 1,
        "addition_shape": 1,
        "known_or_excluded": 2,
        "duplicate_plan": 1,
    }


def test_literal_additions_only_known_values_and_all_history_filtered():
    layout = layout_for()
    excluded = [f"miss{index}.example.test" for index in range(500)]
    excluded[0] = "api.stage.gateway.example.test"
    raw = response(layout, ["dev", "stage", "staging", "staging"])
    candidates, details = expand_vocabulary_response(
        raw, (layout,), TARGET, 20, exclusions=excluded
    )
    assert [item["value"] for item in candidates] == ["api.staging.gateway.example.test"]
    assert details["rejections"]["known_or_excluded"] == 2


def test_layouts_round_robin_without_cartesian_product_and_obey_global_count():
    seeds = SEEDS + ["www.dev.store.example.test", "www.prod.store.example.test"]
    one = layout_for(seeds, "api.{value}.gateway")
    two = layout_for(seeds, "www.{value}.store")
    raw = json.dumps(
        {
            "additions": [
                {"plan_id": layout.plan_id, "values": ["stage", "qa", "uat"]}
                for layout in (one, two)
            ]
        }
    )
    candidates, details = expand_vocabulary_response(raw, (one, two), TARGET, 3)
    assert [item["value"] for item in candidates] == [
        "api.stage.gateway.example.test",
        "www.stage.store.example.test",
        "api.qa.gateway.example.test",
    ]
    assert details["rejections"]["global_limit"] == 3
    assert all(item["status"] == "hypothesis" for item in candidates)


def test_overlong_rendering_is_rejected_without_altering_a_literal():
    prefix = "a" * 60
    layout = layout_for(
        [f"{prefix}-a.example.test", f"{prefix}-b.example.test"], prefix + "-{value}"
    )
    candidates, details = expand_vocabulary_response(
        response(layout, ["long"]), (layout,), TARGET, 20
    )
    assert candidates == [] and details["rejections"]["invalid_rendered_name"] == 1


def test_local_template_overlap_deduplicates_full_names():
    seeds = ["api-dev.example.test", "api-prod.example.test", "www-dev.example.test"]
    one = layout_for(seeds, "api-{value}")
    two = layout_for(seeds, "{value}")
    raw = json.dumps(
        {
            "additions": [
                {"plan_id": one.plan_id, "values": ["stage"]},
                {"plan_id": two.plan_id, "values": ["api-stage"]},
            ]
        }
    )
    candidates, details = expand_vocabulary_response(raw, (one, two), TARGET, 20)
    assert [item["value"] for item in candidates] == ["api-stage.example.test"]
    assert details["rejections"]["duplicate_candidate"] == 1


@pytest.mark.parametrize("count", [0, True, -1, 201, "20", 1.5])
def test_count_bounds_are_strict_before_request_or_parsing(count):
    with pytest.raises(ValueError):
        vocabulary_prompt((layout_for(),), TARGET, count)
    with pytest.raises(ValueError):
        expand_vocabulary_response('{"additions":[]}', (layout_for(),), TARGET, count)


def test_total_observed_and_model_choices_are_capped():
    layout = layout_for()
    candidates, details = expand_vocabulary_response(
        response(layout, [f"v{index}" for index in range(16)]), (layout,), TARGET, 200
    )
    assert candidates == [] and details["rejections"]["choice_limit"] == 1
    raw = json.dumps({"additions": [{"plan_id": layout.plan_id, "values": ["stage"]}] * 25})
    assert expand_vocabulary_response(raw, (layout,), TARGET, 200)[1]["parse_error"]


@pytest.mark.parametrize(
    "excluded",
    ["x.example.test", ["elsewhere.test"], [None], ["x.example.test"] * (MAX_EXCLUSIONS + 1)],
)
def test_bad_or_unbounded_exclusions_are_rejected(excluded):
    with pytest.raises(ValueError):
        vocabulary_exclusions(excluded, TARGET)


def test_exclusion_limit_matches_the_fifty_thousand_node_scan_bound():
    assert MAX_EXCLUSIONS == 50_000
    assert vocabulary_exclusions(["x.example.test"] * MAX_EXCLUSIONS, TARGET) == ("x.example.test",)
