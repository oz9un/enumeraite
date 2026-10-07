#!/usr/bin/env python3
"""Replay production discovery against a frozen, incomplete DNS inventory.

Run separately with PYTHONPATH pointing at a frozen baseline or current src.
Only names requested by DiscoverySession are revealed by the replay validator.
Unknown names remain inconclusive, never invented NXDOMAIN observations.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from enumeraite.bench.dataset import BenchCase, _seeded_split
from enumeraite.bench.metrics import score_run
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.prompt_spec import SUBDOMAIN_SPEC
from enumeraite.core.scan_input import validate_seed_depth


class ArchivedDNS:
    """An offline oracle, not a claim about current DNS or complete coverage."""

    def __init__(self, observations: dict[str, dict]) -> None:
        self.observations = observations
        self.queries: list[str] = []

    def validate(self, values, *, deadline=None):
        results = []
        for value in values:
            self.queries.append(value)
            observation = self.observations.get(value, {})
            status = observation.get("status")
            results.append(
                DNSResult(
                    subdomain=value,
                    exists=status == "verified",
                    ip_addresses=observation.get("addresses", []) if status == "verified" else [],
                    response_time=0,
                    error=(
                        None
                        if status in {"verified", "miss", "wildcard"}
                        else "Unobserved in archived DNS inventory"
                    ),
                    wildcard=status == "wildcard",
                )
            )
        return results


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hash() -> str:
    import enumeraite

    root = Path(enumeraite.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def freeze(args) -> None:
    records = load_json(args.manifest)
    observations: dict[str, dict[str, dict]] = {}
    sources = []
    for record in records:
        path = Path(record["path"])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if record.get("sha256") and record["sha256"] != digest:
            raise ValueError(f"Source changed: {path}")
        data = json.loads(raw)
        target = data["target"]
        target_observations = observations.setdefault(target, {})
        nodes = data.get("nodes", [])
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
        for node in nodes:
            value, status = node.get("value"), node.get("status")
            if not isinstance(value, str) or not value.endswith("." + target):
                continue
            if not SUBDOMAIN_SPEC.validator(value) or status not in {
                "verified",
                "miss",
                "wildcard",
                "inconclusive",
            }:
                continue
            if status == "verified":
                addresses = node.get("addresses", [])
                try:
                    valid_addresses = bool(addresses) and all(
                        ipaddress.ip_address(address) for address in addresses
                    )
                except (ValueError, TypeError):
                    valid_addresses = False
                wildcard = node.get("dns_evidence", {}).get("wildcard", {}).get("status")
                if not (
                    node.get("attempts", 0) > 0
                    and valid_addresses
                    and not node.get("error")
                    and wildcard in {None, "not_detected", "different_answer"}
                ):
                    raise ValueError(f"Unsubstantiated positive observation in {path}")
            previous = target_observations.get(value)
            # Historical positives win over transient later misses. This is
            # explicitly historical recovery, not a present-tense live truth set.
            if previous and previous["status"] == "verified":
                continue
            if previous and status == "inconclusive":
                continue
            target_observations[value] = {
                "status": status,
                "addresses": node.get("addresses", []) if status == "verified" else [],
                "source": node.get("source", "unknown"),
            }
        sources.append({"path": str(path), "sha256": digest, "target": target})
    cases = []
    for target, inventory in sorted(observations.items()):
        truth = sorted(
            value for value, observation in inventory.items() if observation["status"] == "verified"
        )
        if len(truth) < 4:
            continue
        partition = "test" if target in args.test_target else "development"
        for repeat in range(args.repeats):
            salt = f"{args.salt}:{target}:{repeat}"
            known, hidden = _seeded_split(truth, args.seed_fraction, salt, max_seed=args.seed_limit)
            case = BenchCase(target, "subdomain", known, hidden, truth, {"split_salt": salt})
            cases.append({"id": f"{target}:{repeat}", "partition": partition, **asdict(case)})
    represented = {case["target"] for case in cases if case["partition"] == "test"}
    if represented != set(args.test_target):
        raise ValueError("Every final-test target needs at least four verified names")
    manifest = {
        "version": 1,
        "kind": "historical-production-session-replay",
        "salt": args.salt,
        "sources": sources,
        "cases": cases,
        "observations": observations,
        "limitations": [
            "Inventories are incomplete and biased toward previous discovery methods.",
            "Unknown candidates are inconclusive, not ground-truth negatives.",
            "Historical positive observations are not fresh DNS validation.",
            "Repeated splits are not independent targets.",
            "Only seed and queried outcomes enter generation; passive and history intake are disabled.",
        ],
    }
    if args.output.exists():
        raise ValueError("Refusing to overwrite a frozen dataset")
    save_json(args.output, manifest)
    print(
        json.dumps(
            {
                "dataset": str(args.output),
                "cases": len(cases),
                "targets": {
                    target: dict(Counter(x["status"] for x in inventory.values()))
                    for target, inventory in observations.items()
                },
            },
            sort_keys=True,
        )
    )


def replay(args) -> None:
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    if dataset.get("version") != 1:
        raise ValueError("Unsupported frozen dataset version")
    if args.partition == "test" and not args.frozen_source_sha256:
        raise ValueError("Final test requires a previously frozen source SHA-256")
    code_hash = source_hash()
    if args.frozen_source_sha256 and code_hash != args.frozen_source_sha256:
        raise ValueError("Source differs from frozen final-test selection")
    budgets = sorted(set(args.budgets))
    if not budgets or not all(1 <= value <= 5000 for value in budgets):
        raise ValueError("Candidate budgets must be between 1 and 5000")
    result = {
        "version": 1,
        "label": args.label,
        "source_sha256": code_hash,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partition": args.partition,
        "budgets": budgets,
        "limitations": dataset["limitations"],
        "cases": [],
    }
    selected = [
        case
        for case in dataset["cases"]
        if case["partition"] == args.partition
        and (not args.target or case["target"] in args.target)
    ]
    if not selected:
        raise ValueError("No matching benchmark cases")
    for spec in selected:
        validate_seed_depth(spec["target"], spec["known"], args.max_depth)
    for spec in selected:
        case = BenchCase(
            **{key: spec[key] for key in ("target", "kind", "known", "hidden", "universe", "meta")}
        )
        validator = ArchivedDNS(dataset["observations"][case.target])
        options = ScanOptions(
            passive=False,
            max_candidates=len(case.known) + max(budgets),
            max_rounds=0,
            time_limit=args.case_seconds,
            batch_size=40,
            max_depth=args.max_depth,
            max_number=args.max_number,
            stale_rounds=0,
        )
        started = time.monotonic()
        session = DiscoverySession(case.target, options, validator)
        summary = session.run(case.known)
        known = set(case.known)
        candidates = list(dict.fromkeys(value for value in validator.queries if value not in known))
        score = score_run(
            case, candidates, budgets=budgets, requested=max(budgets), trace_limit=max(budgets)
        )
        nodes_by_value = {node["value"]: node for node in session.state["nodes"]}
        row = {
            "id": spec["id"],
            "target": case.target,
            "seed_count": len(case.known),
            "hidden_count": len(case.hidden),
            "options": asdict(options),
            "elapsed_seconds": time.monotonic() - started,
            "attempted_candidates": len(candidates),
            "historical_hits": score.valid,
            "recall_at_budget": score.recall_at_budget,
            "known_hit_rate_lower_bound_at_budget": score.precision_at_budget,
            "stop_reason": summary["stop_reason"],
            "discovery_methods": summary["discovery_methods"],
            "candidate_trace": [
                {
                    key: nodes_by_value[value][key]
                    for key in ("value", "status", "source", "parent", "reason")
                }
                for value in candidates
            ],
        }
        result["cases"].append(row)
        save_json(args.output, result)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "id",
                        "attempted_candidates",
                        "historical_hits",
                        "recall_at_budget",
                        "elapsed_seconds",
                        "stop_reason",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )
    per_target = {}
    for target in sorted({row["target"] for row in result["cases"]}):
        rows = [row for row in result["cases"] if row["target"] == target]
        per_target[target] = {
            budget: sum(row["recall_at_budget"][budget] for row in rows) / len(rows)
            for budget in budgets
        }
    result["macro_recall_at_budget"] = {
        budget: sum(row[budget] for row in per_target.values()) / len(per_target)
        for budget in budgets
    }
    result["per_target_recall_at_budget"] = per_target
    result["target_count"] = len(per_target)
    save_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "macro_recall_at_budget": result["macro_recall_at_budget"],
                "target_count": len(per_target),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("freeze")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--test-target", action="append", default=[], required=True)
    build.add_argument("--seed-fraction", type=float, default=0.35)
    build.add_argument("--seed-limit", type=int, default=100)
    build.add_argument("--repeats", type=int, default=5)
    build.add_argument("--salt", required=True)
    build.set_defaults(function=freeze)
    run = commands.add_parser("replay")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--partition", choices=("development", "test"), required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--target", action="append", default=[])
    run.add_argument("--budgets", nargs="+", type=int, default=[100, 250, 500, 1000, 2500])
    run.add_argument("--case-seconds", type=float, default=120)
    run.add_argument("--max-depth", type=int, default=4)
    run.add_argument("--max-number", type=int, default=99)
    run.add_argument("--frozen-source-sha256")
    run.set_defaults(function=replay)
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.set_defaults(function=lambda args: print(source_hash()))
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
