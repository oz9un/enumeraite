"""Offline root-versus-hostname input resolution and explicit scan boundaries."""

from dataclasses import asdict

import pytest
import tldextract

from enumeraite.core import scan_input
from enumeraite.core.scan_input import resolve_scan_input, validate_seed_depth


@pytest.mark.parametrize(
    "target", ["example.com", "example.co.uk", "tenant.github.io", "tenant.appspot.com"]
)
def test_registrable_domain_is_passive_first_without_automatic_seed(target):
    result = resolve_scan_input(target)
    assert asdict(result) == {
        "target": target,
        "seeds": [],
        "input": target,
        "mode": "passive",
        "passive_default": True,
    }


@pytest.mark.parametrize(
    "hostname,target",
    [
        ("ny-01-iphone.example.com", "example.com"),
        ("api.ny-01-iphone.example.com", "example.com"),
        ("ny-01-iphone.example.co.uk", "example.co.uk"),
        ("preview.tenant.github.io", "tenant.github.io"),
        ("api.tenant.appspot.com", "tenant.appspot.com"),
    ],
)
def test_known_hostname_becomes_seed_inside_registrable_or_private_tenant_scope(hostname, target):
    result = resolve_scan_input(hostname)
    assert result.target == target
    assert result.input == hostname
    assert result.seeds == [hostname]
    assert result.mode == "seeded" and not result.passive_default


def test_unicode_case_and_trailing_dot_normalize_before_resolution_and_deduplication():
    result = resolve_scan_input("API.BÜCHER.DE.", seeds=["api.xn--bcher-kva.de", "WWW.BÜCHER.DE."])
    assert result.input == "api.xn--bcher-kva.de"
    assert result.target == "xn--bcher-kva.de"
    assert result.seeds == ["api.xn--bcher-kva.de", "www.xn--bcher-kva.de"]


@pytest.mark.parametrize("suffix", ["com", "co.uk", "github.io", "appspot.com"])
def test_naked_public_or_private_suffix_is_not_a_scan_target_or_scope(suffix):
    with pytest.raises(ValueError):
        resolve_scan_input(suffix)
    with pytest.raises(ValueError):
        resolve_scan_input("host." + suffix, scope=suffix)


def test_explicit_scope_pins_narrower_zone_and_auto_seed_stays_inside_it():
    result = resolve_scan_input("ny-01-iphone.ops.example.com", scope="OPS.EXAMPLE.COM.")
    assert result.target == "ops.example.com"
    assert result.seeds == ["ny-01-iphone.ops.example.com"]
    assert result.mode == "seeded"
    exact = resolve_scan_input("ops.example.com", scope="ops.example.com")
    assert exact.target == exact.input == "ops.example.com"
    assert exact.seeds == [] and exact.passive_default


@pytest.mark.parametrize(
    "value,scope",
    [
        ("example.com", "ops.example.com"),
        ("api.example.com", "example.net"),
        ("example.com.attacker.test", "example.com"),
        ("api.tenant.github.io", "other.github.io"),
    ],
)
def test_explicit_scope_must_contain_input_at_a_dns_label_boundary(value, scope):
    with pytest.raises(ValueError, match="within"):
        resolve_scan_input(value, scope=scope)


def test_unknown_suffix_never_guesses_a_wider_boundary():
    result = resolve_scan_input("ny-01-iphone.example.internal")
    assert result.target == result.input == "ny-01-iphone.example.internal"
    assert result.seeds == [] and result.passive_default
    explicit = resolve_scan_input("ny-01-iphone.example.internal", scope="example.internal")
    assert explicit.target == "example.internal"
    assert explicit.seeds == ["ny-01-iphone.example.internal"]
    assert not explicit.passive_default


def test_explicit_seeds_switch_root_to_seeded_mode_without_changing_scope():
    result = resolve_scan_input(
        "example.com", seeds=["NY-01-IPHONE.EXAMPLE.COM.", "ny-01-iphone.example.com"]
    )
    assert result.target == "example.com"
    assert result.seeds == ["ny-01-iphone.example.com"]
    assert result.mode == "seeded" and not result.passive_default


@pytest.mark.parametrize(
    "seed",
    [
        "example.com",
        "api.example.net",
        "example.com.evil.test",
        "*.example.com",
        "https://api.example.com",
        "127.0.0.1",
    ],
)
def test_invalid_or_outside_explicit_seeds_fail_instead_of_being_silently_ignored(seed):
    with pytest.raises(ValueError):
        resolve_scan_input("example.com", seeds=[seed])


@pytest.mark.parametrize("seeds", ["api.example.com", ("api.example.com",), {}, True, [None]])
def test_seed_collection_must_be_a_list_of_hostname_strings(seeds):
    with pytest.raises(ValueError):
        resolve_scan_input("example.com", seeds=seeds)


def test_seed_limit_includes_automatic_input_and_deduplicates_that_input():
    seeds = [f"host{index}.example.com" for index in range(1000)]
    assert len(resolve_scan_input("host0.example.com", seeds=seeds).seeds) == 1000
    with pytest.raises(ValueError, match="1000"):
        resolve_scan_input("another.example.com", seeds=seeds)
    with pytest.raises(ValueError, match="1000"):
        resolve_scan_input("example.com", seeds=seeds + [seeds[0]])


def test_first_resolution_reads_only_bundled_suffix_list_without_network_or_cache(
    monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        pytest.fail("Scope inference must never make a network request")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("socket.getaddrinfo", forbidden)
    monkeypatch.setenv("TLDEXTRACT_CACHE", str(tmp_path))
    monkeypatch.setattr(
        scan_input,
        "_EXTRACT",
        tldextract.TLDExtract(
            suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
        ),
    )
    assert resolve_scan_input("api.tenant.github.io").target == "tenant.github.io"
    assert resolve_scan_input("ny-01.example.co.uk").target == "example.co.uk"
    assert list(tmp_path.iterdir()) == []


def test_seed_depth_guard_reports_required_depth_and_accepts_increased_depth_or_narrower_scope():
    seed = "api.dev.us.east.cluster.example.com"
    with pytest.raises(ValueError, match="requires 5, configured 4"):
        validate_seed_depth("example.com", [seed], 4)
    validate_seed_depth("example.com", [seed], 5)
    validate_seed_depth("cluster.example.com", [seed], 4)
    validate_seed_depth("example.com", [], 4)


@pytest.mark.parametrize("depth", [0, 11, True, 4.5, "4"])
def test_seed_depth_guard_requires_valid_numeric_limit(depth):
    with pytest.raises(ValueError, match="integer"):
        validate_seed_depth("example.com", ["api.example.com"], depth)


def test_seed_depth_guard_rejects_outside_names():
    with pytest.raises(ValueError, match="within the scan scope"):
        validate_seed_depth("example.com", ["api.example.net"], 4)
