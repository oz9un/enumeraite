import pytest

from enumeraite.core.seeds import bounded_seed_sample, normalize_seed, read_seed_file, valid_seed


def test_seed_sample_is_deterministic_spread_and_bounded():
    values = [f"/item/{index:03d}" for index in range(100)]
    sampled = bounded_seed_sample(values, limit=5, char_limit=1000)
    assert sampled == [
        "/item/000",
        "/item/024",
        "/item/049",
        "/item/074",
        "/item/099",
    ]
    assert bounded_seed_sample(values, limit=5, char_limit=25) == [
        "/item/000",
        "/item/024",
    ]


def test_seed_sample_deduplicates_and_rejects_invalid_budgets():
    assert bounded_seed_sample(["/one", "/one", "", "/two"]) == ["/one", "/two"]
    with pytest.raises(ValueError, match="positive"):
        bounded_seed_sample(["/one"], limit=0)


def test_seed_sample_rejects_ambiguous_or_unbounded_iterables():
    with pytest.raises(ValueError, match="iterable of seed strings"):
        bounded_seed_sample("/one")
    with pytest.raises(ValueError, match="only strings"):
        bounded_seed_sample(["/one", 2])
    with pytest.raises(ValueError, match="100000-item"):
        bounded_seed_sample((f"/{index}" for index in range(100_001)))


def test_seed_file_reader_enforces_byte_line_and_item_limits(tmp_path):
    valid = tmp_path / "valid.txt"
    valid.write_text("# comment\n/one\n\n/two\n", encoding="utf-8")
    assert read_seed_file(valid) == ["/one", "/two"]

    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"x" * 11)
    with pytest.raises(ValueError, match="byte limit"):
        read_seed_file(oversized, max_bytes=10)

    too_many = tmp_path / "too-many.txt"
    too_many.write_text("/one\n/two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="item limit"):
        read_seed_file(too_many, max_items=1)


def test_path_evidence_allows_safe_templates_but_not_output_escapes():
    assert valid_seed("/users/{userId}/profile", "path")
    assert valid_seed("/users/:userId/profile", "path")
    assert not valid_seed("/users/{id}?debug=1", "path")
    assert not valid_seed("/../../etc", "path")
    assert normalize_seed("API.Example.COM.", "subdomain") == "api.example.com"
