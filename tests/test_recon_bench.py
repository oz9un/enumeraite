import json
from argparse import Namespace

import pytest

from enumeraite.core.discovery_session import DiscoverySession, ScanOptions
from tools.recon_bench import ArchivedDNS, replay


def test_unobserved_dns_is_inconclusive_not_a_negative():
    oracle = ArchivedDNS({"missing.example.com": {"status": "miss"}})
    session = DiscoverySession(
        "example.com",
        ScanOptions(passive=False, max_candidates=2, max_rounds=1, time_limit=5),
        oracle,
    )
    session.run(["missing.example.com", "unobserved.example.com"])
    assert {node["value"]: node["status"] for node in session.state["nodes"]} == {
        "missing.example.com": "miss",
        "unobserved.example.com": "inconclusive",
    }


@pytest.fixture
def replay_args(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "version": 1,
                "limitations": [],
                "observations": {"example.com": {}},
                "cases": [
                    {
                        "id": "nested-visible-seed",
                        "target": "example.com",
                        "partition": "test",
                        "kind": "subdomain",
                        "known": ["deep.api.example.com"],
                        "hidden": ["other.example.com"],
                        "universe": ["deep.api.example.com", "other.example.com"],
                        "meta": {},
                    }
                ],
            }
        )
    )
    return Namespace(
        dataset=dataset,
        partition="test",
        label="test",
        output=tmp_path / "result.json",
        target=[],
        budgets=[10],
        case_seconds=1,
        max_depth=2,
        max_number=10,
        frozen_source_sha256=None,
    )


def test_final_test_refuses_unfrozen_or_mismatched_source(replay_args):
    for fingerprint in (None, "0" * 64):
        replay_args.frozen_source_sha256 = fingerprint
        with pytest.raises(ValueError):
            replay(replay_args)
        assert not replay_args.output.exists()


def test_depth_rejection_cannot_silently_change_seed_budget(replay_args):
    replay_args.partition = "development"
    replay_args.max_depth = 1
    dataset = json.loads(replay_args.dataset.read_text())
    dataset["cases"][0]["partition"] = "development"
    replay_args.dataset.write_text(json.dumps(dataset))
    with pytest.raises(ValueError):
        replay(replay_args)
    assert not replay_args.output.exists()
