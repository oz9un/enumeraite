"""End-to-end scan CLI flows using real sessions and fake external services."""

import importlib
import json
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from enumeraite.cli.main import cli
from enumeraite.cli.scan_tree import write_tree
from enumeraite.core.config import Config
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.passive import PassiveResult

scan_module = importlib.import_module("enumeraite.cli.scan_command")


@pytest.fixture
def services(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        names=["ny-01-iphone.example.com"],
        hits={
            "ny-01-iphone.example.com",
            "ny-02-iphone.example.com",
            "api.ny-02-iphone.example.com",
        },
        wildcard=set(),
        errors=[],
        truncated=False,
        passive_calls=[],
        dns_calls=[],
        events=[],
        dns_options=[],
    )

    def passive(target, **options):
        fake.events.append("passive")
        fake.passive_calls.append((target, options))
        return PassiveResult(
            names=list(fake.names),
            sources={"test-passive": list(fake.names)},
            errors=list(fake.errors),
            truncated=fake.truncated,
        )

    class FakeDNS:
        def __init__(self, target, **options):
            fake.dns_options.append((target, options))
            self.query_count = 0

        def validate(self, values, *, deadline):
            fake.events.append("dns")
            fake.dns_calls.append(list(values))
            self.query_count += len(values)
            return [
                DNSResult(
                    value,
                    exists=value in fake.hits,
                    ip_addresses=["192.0.2.25"] if value in fake.hits else [],
                    response_time=0.001,
                    wildcard=value in fake.wildcard,
                )
                for value in values
            ]

    monkeypatch.setattr(scan_module, "collect_passive", passive)
    monkeypatch.setattr(scan_module, "ScanDNS", FakeDNS)
    monkeypatch.setattr(scan_module, "load_config", lambda: Config())
    # Explicitly simulate no available provider without probing localhost or
    # reading the developer's configured API credentials.
    monkeypatch.setattr(scan_module, "detect_provider", lambda config: None)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "application-state"))
    return fake


def invoke(tmp_path, *arguments, input=None, target="example.com"):
    return CliRunner().invoke(
        cli,
        [
            "scan",
            target,
            "--state",
            str(tmp_path / "scan.json"),
            "--time",
            "10s",
            "--max-candidates",
            "100",
            "--rounds",
            "2",
            *arguments,
        ],
        input=input,
    )


def test_scan_is_listed_and_help_explains_verified_goal():
    main = CliRunner().invoke(cli, ["--help"])
    help_result = CliRunner().invoke(cli, ["scan", "--help"])
    assert main.exit_code == help_result.exit_code == 0
    assert "scan" in main.stdout
    assert "--find" in help_result.stdout
    assert "verified subdomains" in help_result.stdout
    assert "--budget-usd" in help_result.stdout
    assert "--resume" in help_result.stdout


@pytest.mark.parametrize(
    "flags,stale,rounds",
    [([], 0, 10), (["--until-stale"], 2, 100), (["--until-stale", "--rounds", "3"], 2, 3)],
)
def test_broad_default_and_explicit_stale_policy(services, tmp_path, flags, stale, rounds):
    state_path = tmp_path / "policy.json"
    result = CliRunner().invoke(
        cli,
        ["scan", "example.com", "--state", str(state_path), "--find", "1", *flags],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads(state_path.read_text())
    assert saved["options"]["stale_rounds"] == stale
    assert saved["options"]["max_rounds"] == rounds


@pytest.mark.parametrize(
    "flags,enabled",
    [([], False), (["--vocabulary-breadth"], True), (["--no-vocabulary-breadth"], False)],
)
def test_vocabulary_breadth_is_opt_in_and_saved(services, tmp_path, flags, enabled):
    result = invoke(tmp_path, "--find", "1", *flags)
    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "scan.json").read_text())
    assert saved["options"]["vocabulary_breadth"] is enabled
    assert saved["provider"] is None
    assert saved["ledger"] is None


def test_vocabulary_breadth_resume_preserves_setting_until_explicitly_disabled(services, tmp_path):
    initial = invoke(tmp_path, "--find", "1", "--vocabulary-breadth")
    assert initial.exit_code == 0, initial.output
    resumed = invoke(tmp_path, "--resume", "--find", "2")
    assert resumed.exit_code == 0, resumed.output
    saved = json.loads((tmp_path / "scan.json").read_text())
    assert saved["options"]["vocabulary_breadth"] is True
    assert len(services.passive_calls) == 1

    disabled = invoke(tmp_path, "--resume", "--no-vocabulary-breadth")
    assert disabled.exit_code == 0, disabled.output
    saved = json.loads((tmp_path / "scan.json").read_text())
    assert saved["options"]["vocabulary_breadth"] is False
    assert len(services.passive_calls) == 1


def test_no_key_scan_collects_known_names_then_recursively_expands(services, tmp_path):
    result = invoke(tmp_path, "--find", "3")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "ny-01-iphone.example.com",
        "ny-02-iphone.example.com",
        "api.ny-02-iphone.example.com",
    ]
    assert services.events[0] == "passive"
    assert services.dns_calls[0][0] == "ny-01-iphone.example.com"
    state = json.loads((tmp_path / "scan.json").read_text())
    nodes = {node["value"]: node for node in state["nodes"]}
    assert nodes["ny-01-iphone.example.com"]["source"] == "passive"
    assert nodes["ny-02-iphone.example.com"]["parent"] == "ny-01-iphone.example.com"
    assert nodes["api.ny-02-iphone.example.com"]["parent"] == "ny-02-iphone.example.com"
    assert state["stop_reason"] == "verified target reached"
    assert state["provider"] is None
    assert "no paid AI calls" in result.stderr
    assert "Collected 1 known names" in result.stderr
    assert "Scanning" not in result.stdout


def test_json_is_parseable_without_progress_and_records_provenance(services, tmp_path):
    result = invoke(tmp_path, "--json", "--find", "2")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["target"] == "example.com"
    assert report["verified"] == 2
    assert report["known_verified"] == 1
    assert report["findings"] == ["ny-01-iphone.example.com", "ny-02-iphone.example.com"]
    assert report["attempted"] > report["verified"]
    assert report["passive_sources"] == {"test-passive": ["ny-01-iphone.example.com"]}
    assert report["patterns"][0]["slots"][1]["padding"] == 2
    assert report["state_file"] == str(tmp_path / "scan.json")
    assert "Finished:" in result.stderr


def test_no_passive_bootstraps_without_any_seed_and_find_counts_hits(services, tmp_path):
    services.hits = {"www.example.com", "api.example.com"}
    result = invoke(tmp_path, "--no-passive", "--provider", "none", "--find", "2", "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert services.passive_calls == []
    assert report["findings"] == ["www.example.com", "api.example.com"]
    assert report["verified"] == 2
    assert report["attempted"] == 3  # mail was attempted and missed.
    assert report["known_verified"] == 0
    assert report["stop_reason"] == "verified target reached"


@pytest.mark.parametrize("goal", [2, 50_001])
def test_candidate_cap_can_stop_before_verified_goal(services, tmp_path, goal):
    services.hits = {"www.example.com", "api.example.com"}
    result = invoke(
        tmp_path, "--no-passive", "--find", str(goal), "--max-candidates", "1", "--json"
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["verified"] == report["attempted"] == 1
    assert report["stop_reason"] == "candidate limit reached"
    assert report["findings"] == ["www.example.com"]


def test_passive_failures_and_partial_results_are_visible_but_do_not_block(services, tmp_path):
    services.errors = ["one public source timed out"]
    services.truncated = True
    result = invoke(tmp_path, "--find", "1", "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["verified"] == 1
    assert report["passive_truncated"] is True
    assert "one public source timed out" in report["warnings"]
    assert "partial view" in result.stderr


def test_unverified_passive_names_and_wildcards_are_not_findings(services, tmp_path):
    services.names += ["gone.example.com", "wild.example.com", "outside.invalid"]
    services.hits.add("wild.example.com")
    services.wildcard.add("wild.example.com")
    result = invoke(tmp_path, "--json", "--max-candidates", "25")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    nodes = {node["value"]: node for node in report["nodes"]}
    assert nodes["gone.example.com"]["status"] == "miss"
    assert nodes["wild.example.com"]["status"] == "wildcard"
    assert "gone.example.com" not in report["findings"]
    assert "wild.example.com" not in report["findings"]
    assert "outside.invalid" not in nodes
    assert "outside.invalid" not in [value for batch in services.dns_calls for value in batch]


def test_stdin_seeds_are_normalized_scoped_and_deduplicated(services, tmp_path):
    seed_text = "# exported inventory\nNY-01-IPHONE.EXAMPLE.COM.\nny-01-iphone.example.com\n"
    result = invoke(
        tmp_path, "--seeds", "-", "--no-passive", "--find", "1", "--quiet", input=seed_text
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == "ny-01-iphone.example.com\n"
    assert result.stderr == ""
    assert services.passive_calls == []
    state = json.loads((tmp_path / "scan.json").read_text())
    assert state["nodes"][0]["source"] == "seed"
    assert len([node for node in state["nodes"] if node["source"] == "seed"]) == 1


def test_out_of_scope_stdin_seed_rejected_before_any_network(services, tmp_path):
    result = invoke(tmp_path, "--seeds", "-", input="api.example.com\nother.invalid\n")
    assert result.exit_code != 0
    assert "Every seed must be a subdomain within" in result.stderr
    assert services.passive_calls == [] and services.dns_calls == []


def test_hostname_input_seeds_scope_and_finds_sibling_under_tiny_candidate_cap(services, tmp_path):
    result = invoke(
        tmp_path,
        "--find",
        "2",
        "--max-candidates",
        "2",
        "--max-number",
        "2",
        "--json",
        target="NY-01-IPHONE.EXAMPLE.COM.",
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["target"] == "example.com"
    assert report["findings"] == ["ny-01-iphone.example.com", "ny-02-iphone.example.com"]
    assert services.passive_calls == []
    assert services.dns_calls == [["ny-01-iphone.example.com"], ["ny-02-iphone.example.com"]]
    assert report["scan_input"]["mode"] == "seeded"
    assert report["scan_input"]["input"] == "ny-01-iphone.example.com"
    assert all(node["source"] != "bootstrap" for node in report["nodes"])


def test_repeatable_inline_and_file_seeds_merge_normalize_and_disable_passive_by_default(
    services, tmp_path
):
    seed_file = tmp_path / "inventory.txt"
    seed_file.write_text("ny-01-iphone.example.com\ncdn.example.com\n")
    result = invoke(
        tmp_path,
        "--seed",
        "NY-01-IPHONE.EXAMPLE.COM.",
        "--seed",
        "api.example.com",
        "--seeds",
        str(seed_file),
        "--json",
        "--find",
        "1",
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["scan_input"]["seeds"] == [
        "ny-01-iphone.example.com",
        "api.example.com",
        "cdn.example.com",
    ]
    assert services.passive_calls == []
    assert report["findings"] == ["ny-01-iphone.example.com"]


def test_explicit_passive_can_collect_more_names_for_hostname_input(services, tmp_path):
    result = invoke(
        tmp_path, "--passive", "--find", "1", "--json", target="ny-01-iphone.example.com"
    )
    assert result.exit_code == 0, result.output
    assert services.passive_calls[0][0] == "example.com"
    assert json.loads((tmp_path / "scan.json").read_text())["options"]["passive"] is True


def test_explicit_scope_keeps_sibling_discovery_within_narrower_zone(services, tmp_path):
    first = "ny-01-iphone.ops.example.com"
    second = "ny-02-iphone.ops.example.com"
    services.hits = {first, second}
    result = invoke(
        tmp_path,
        "--scope",
        "ops.example.com",
        "--max-candidates",
        "2",
        "--max-number",
        "2",
        "--find",
        "2",
        "--json",
        target=first,
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["target"] == "ops.example.com"
    assert report["findings"] == [first, second]
    assert services.dns_options[0][0] == "ops.example.com"
    assert services.passive_calls == []


def test_explicit_state_resume_preserves_legacy_exact_hostname_scope(services, tmp_path):
    narrow = "ops.example.com"
    services.names = ["api." + narrow, "cdn." + narrow]
    services.hits = set(services.names)
    first = invoke(tmp_path, "--scope", narrow, "--find", "1", target=narrow)
    assert first.exit_code == 0, first.output
    resumed = invoke(tmp_path, "--resume", "--find", "2", "--json", target=narrow)
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.stdout)["target"] == narrow
    assert services.dns_options[-1][0] == narrow
    assert len(services.passive_calls) == 1


def test_default_resume_prefers_legacy_exact_scope_checkpoint(services, tmp_path):
    runner = CliRunner()
    narrow = "ops.example.com"
    services.names = ["api." + narrow, "cdn." + narrow]
    services.hits = set(services.names)
    first = runner.invoke(cli, ["scan", narrow, "--scope", narrow, "--find", "1", "--quiet"])
    assert first.exit_code == 0, first.output
    resumed = runner.invoke(cli, ["scan", narrow, "--resume", "--find", "2", "--json", "--quiet"])
    assert resumed.exit_code == 0, resumed.output
    report = json.loads(resumed.stdout)
    assert report["target"] == narrow
    assert f"/{narrow}/" in report["state_file"]


def test_resume_cannot_widen_saved_scope_even_when_parent_domain_matches(services, tmp_path):
    narrow = "ops.example.com"
    services.names = ["api." + narrow]
    services.hits = set(services.names)
    assert invoke(tmp_path, "--scope", narrow, "--find", "1", target=narrow).exit_code == 0
    before = list(services.dns_calls)
    resumed = invoke(tmp_path, "--resume", "--scope", "example.com", target=narrow)
    assert resumed.exit_code != 0
    assert "Checkpoint target does not match" in resumed.stderr
    assert services.dns_calls == before


def test_deep_hostname_must_fit_depth_limit_before_any_dns_or_passive_request(services, tmp_path):
    seed = "api.dev.us.east.cluster.example.com"
    result = invoke(tmp_path, "--find", "1", target=seed)
    assert result.exit_code != 0
    assert "Starting subdomain exceeds max-depth" in result.stderr
    assert "requires 5, configured 4" in result.stderr
    assert "increase depth or narrow scope" in result.stderr
    assert services.dns_calls == services.passive_calls == []


def test_deep_hostname_works_when_user_explicitly_increases_depth(services, tmp_path):
    seed = "api.dev.us.east.cluster.example.com"
    services.hits = {seed}
    result = invoke(tmp_path, "--find", "1", "--max-depth", "5", "--json", target=seed)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["findings"] == [seed]
    assert services.dns_calls == [[seed]]
    assert services.passive_calls == []


@pytest.mark.parametrize(
    "limits",
    [
        [],
        ["--max-candidates", "0", "--rounds", "0", "--time", "0"],
        ["--max-candidates", "50001", "--rounds", "101", "--time", "25h"],
    ],
)
def test_resume_continues_saved_queue_without_recollecting_or_rechecking(
    services, tmp_path, limits
):
    initial = invoke(tmp_path, "--find", "1", "--quiet", *limits)
    assert initial.exit_code == 0, initial.output
    assert services.dns_calls == [["ny-01-iphone.example.com"]]
    resumed = CliRunner().invoke(
        cli,
        [
            "scan",
            "example.com",
            "--state",
            str(tmp_path / "scan.json"),
            "--resume",
            "--find",
            "2",
            "--json",
            "--quiet",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    report = json.loads(resumed.stdout)
    assert report["verified"] == 2
    assert len(services.passive_calls) == 1
    attempted = [value for batch in services.dns_calls for value in batch]
    assert len(attempted) == len(set(attempted))
    assert "ny-02-iphone.example.com" in report["findings"]
    assert resumed.stderr == ""


def test_default_checkpoint_can_be_resumed_without_a_file_argument(services, tmp_path):
    runner = CliRunner()
    initial = runner.invoke(cli, ["scan", "example.com", "--find", "1", "--quiet"])
    assert initial.exit_code == 0, initial.output
    paths = list((tmp_path / "application-state").rglob("scan-*.json"))
    assert len(paths) == 1
    resumed = runner.invoke(
        cli, ["scan", "example.com", "--resume", "--find", "2", "--json", "--quiet"]
    )
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.stdout)["verified"] == 2
    assert len(services.passive_calls) == 1


def test_resume_rejects_a_different_target_before_network_calls(services, tmp_path):
    assert invoke(tmp_path, "--find", "1").exit_code == 0
    before = len(services.dns_calls)
    result = invoke(tmp_path, "--resume", target="different.example")
    assert result.exit_code != 0
    assert "target does not match" in result.stderr
    assert len(services.dns_calls) == before
    assert len(services.passive_calls) == 1


def test_output_and_tree_files_leave_stdout_clean(services, tmp_path):
    output = tmp_path / "findings.txt"
    tree = tmp_path / "tree.html"
    result = invoke(
        tmp_path, "--find", "2", "--output", str(output), "--tree", "--tree-file", str(tree)
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert output.read_text().splitlines() == [
        "ny-01-iphone.example.com",
        "ny-02-iphone.example.com",
    ]
    assert "├──" in result.stderr or "└──" in result.stderr
    assert "ny-02-iphone.example.com" in result.stderr
    content = tree.read_text()
    assert "ny-02-iphone.example.com" in content
    assert "&lt;number:02d&gt;" in content
    assert 'id="filter"' in content
    assert 'src="http' not in content


def test_output_cannot_overwrite_checkpoint_or_seed_input(services, tmp_path):
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("ny-01-iphone.example.com\n")
    original = seeds.read_text()
    result = invoke(tmp_path, "--seeds", str(seeds), "--output", str(seeds))
    assert result.exit_code != 0
    assert "must not overwrite" in result.stderr
    assert seeds.read_text() == original
    state = tmp_path / "scan.json"
    result = invoke(tmp_path, "--output", str(state))
    assert result.exit_code != 0
    assert not state.exists()
    assert services.passive_calls == services.dns_calls == []


@pytest.mark.parametrize(
    "target", ["https://example.com", "*.example.com", "localhost", "127.0.0.1"]
)
def test_invalid_targets_fail_before_network(target, services, tmp_path):
    result = invoke(tmp_path, target=target)
    assert result.exit_code != 0
    assert result.stdout == ""
    assert services.passive_calls == services.dns_calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        ("--find", "0"),
        ("--max-candidates", "-1"),
        ("--rounds", "-1"),
        ("--workers", "101"),
        ("--max-depth", "11"),
        ("--rate", "nan"),
        ("--time", "never"),
        ("--time", "-1s"),
        ("--time", "9" * 400),
        ("--budget-usd", "-1"),
        ("--budget-usd", "nan"),
        ("--budget-usd", "inf"),
        ("--input-price", "nan"),
        ("--output-price", "inf"),
    ],
)
def test_invalid_options_fail_before_network(arguments, services, tmp_path):
    result = invoke(tmp_path, *arguments)
    assert result.exit_code != 0
    assert result.stdout == ""
    assert services.passive_calls == services.dns_calls == []


def test_zero_model_budget_still_allows_free_scan(services, tmp_path):
    result = invoke(tmp_path, "--budget-usd", "0", "--find", "1", "--quiet")
    assert result.exit_code == 0, result.output
    assert result.stdout == "ny-01-iphone.example.com\n"


def test_tree_escapes_all_remote_text_and_preserves_only_its_own_script(tmp_path):
    injection = "</script><script>alert('xss')</script><img src=x onerror=alert(1)>"
    report = {
        "target": injection,
        "verified": 1,
        "known_verified": 1,
        "attempted": 1,
        "round": 0,
        "stop_reason": injection,
        "warnings": [injection],
        "inconclusive": 2,
        "nodes": [
            {
                "value": injection,
                "parent": None,
                "status": "verified' onmouseover='alert(1)",
                "source": injection,
                "pattern": injection,
                "reason": injection,
                "round": 0,
            }
        ],
        "patterns": [{"pattern": injection, "examples": [injection]}],
    }
    destination = tmp_path / "tree.html"
    write_tree(report, destination)
    content = destination.read_text()

    class Tags(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []
            self.attributes = []

        def handle_starttag(self, tag, attributes):
            self.tags.append(tag)
            self.attributes.extend(key for key, _ in attributes)

    parsed = Tags()
    parsed.feed(content)
    assert parsed.tags.count("script") == 1
    assert "img" not in parsed.tags
    assert "onerror" not in parsed.attributes
    assert "onmouseover" not in parsed.attributes
    assert injection not in content
    assert "&lt;/script&gt;&lt;script&gt;" in content
    assert "2 DNS checks were inconclusive" in content
    assert 'class="verified-only"' in content
    assert "li[data-status]:not([data-status='verified'])" in content
    assert "Show all attempts" in content
