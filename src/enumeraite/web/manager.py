"""Thread-safe, local-only scan orchestration for the browser dashboard.

The HTTP layer sees immutable snapshots, never a live DiscoverySession or API
key. Credentials are held in memory and captured separately for each run.
"""

from __future__ import annotations

import copy
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from ..core.discovery_metrics import discovery_method_counts
from ..core.discovery_session import (
    DiscoverySession,
    ScanOptions,
    normalize_target,
    validate_checkpoint,
)
from ..core.discovery_vocabulary import VOCABULARY_PROTOCOL_VERSION
from ..core.passive import collect_passive
from ..core.scan_budget import SpendLedger, price_for
from ..core.scan_dns import ScanDNS
from ..core.scan_input import resolve_scan_input, validate_seed_depth
from ..core.scan_state import ScanStore
from ..core.subdomain_patterns import describe_patterns
from ..core.text_safety import safe_model_text
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.openai_compatible import OpenAICompatibleProvider

SCAN_ID = re.compile(r"[a-f0-9]{32}\Z")
MODELS = {
    "minimax": ("MiniMax-M3", "MiniMax-M2.7"),
    "openai": ("gpt-4.1-mini", "gpt-4.1"),
    "anthropic": ("claude-sonnet-4-6",),
}
KEY_PROVIDERS = {*MODELS, "jev"}
ENDPOINTS = {
    "minimax": "https://api.minimax.io/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}
IDENTITIES = {
    "minimax": "openai-compat:minimax",
    "openai": "openai-compat:openai",
    "anthropic": "anthropic",
}
MAX_EVENTS = 300
STATE_FIELDS = {
    "version",
    "target",
    "options",
    "initialized",
    "nodes",
    "round",
    "attempted",
    "stale",
    "elapsed_seconds",
    "stop_reason",
    "warnings",
    "passive_sources",
    "passive_truncated",
    "provider",
    "ledger",
}
PREDICTION_DEFAULTS: dict[str, Any] = {
    "predictor": None,
    "prediction_ledger": None,
    "predictions": [],
    "vocabulary_protocol_version": VOCABULARY_PROTOCOL_VERSION,
}
STATE_FIELDS.update(PREDICTION_DEFAULTS)


def _state_projection(state: dict[str, Any]) -> dict[str, Any]:
    """Canonical optional fields keep pre-prediction checkpoints resumable."""
    return {
        name: copy.deepcopy(state[name] if name in state else PREDICTION_DEFAULTS[name])
        for name in STATE_FIELDS
    }


# Redaction must never rename fixed protocol fields. Passive-source labels are
# dynamic and remain redacted, as do all text values.
SCHEMA_FIELDS = (
    STATE_FIELDS
    | set(ScanOptions.__dataclass_fields__)
    | {
        "_ui",
        "created_at",
        "status",
        "prior_verified",
        "history_enabled",
        "value",
        "parent",
        "source",
        "pattern",
        "reason",
        "depth",
        "expanded_revision",
        "addresses",
        "aliases",
        "dns_evidence",
        "origin_evidence",
        "queued_at",
        "collected_at",
        "query",
        "matched_name",
        "certificate_id",
        "issuer",
        "subject_common_name",
        "first_seen",
        "not_before",
        "not_after",
        "entry_timestamp",
        "address",
        "evidence",
        "confidence",
        "focus",
        "batch",
        "plan_description",
        "layout",
        "word",
        "numeric_ceiling",
        "duration_ms",
        "records",
        "A",
        "AAAA",
        "error",
        "attempts",
        "name",
        "model",
        "limit_usd",
        "spent_usd",
        "reserved_usd",
        "remaining_usd",
        "calls",
        "uncertain_calls",
        "blocked",
        "pricing",
        "input_usd_per_million",
        "output_usd_per_million",
        "cached_input_usd_per_million",
        "checked_at",
        "message",
        "hostname",
        "candidate",
        "query_id",
        "worker_id",
        "query_kind",
        "probe",
        "count",
        "sources",
        "verified",
        "known_verified",
        "discovery_methods",
        "method",
        "candidates",
        "checked",
        "pending",
        "miss",
        "findings",
        "queued",
        "inconclusive",
        "wildcard",
        "stage",
        "plan_id",
        "description",
        "score",
    }
)


def default_storage_dir() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
    return base / "enumeraite" / "ui"


@dataclass
class _Record:
    id: str
    target: str
    options: dict[str, Any]
    provider: str
    model: str | None
    created_at: float
    status: str = "running"
    phase: str = "Starting scan"
    state: dict[str, Any] | None = None
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    pattern_findings: tuple[str, ...] = ()
    active: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: deque = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error: str | None = None
    revision: int = 0
    event_id: int = 0
    snapshot_at: float = field(default_factory=time.monotonic)
    elapsed: float = 0
    predictor: dict[str, str] | None = None
    prior_verified: tuple[str, ...] = ()
    history_enabled: bool = False


class ScanManager:
    """Run one domain session at a time, with concurrent DNS workers underneath."""

    def __init__(
        self,
        storage_dir: Path | str | None = None,
        *,
        session_factory: Callable = DiscoverySession,
        dns_factory: Callable = ScanDNS,
        passive_fetch: Callable = collect_passive,
        provider_factory: Callable | None = None,
        predictor_factory: Callable | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir is not None else default_storage_dir()
        self._session_factory = session_factory
        self._dns_factory = dns_factory
        self._passive_fetch = passive_fetch
        self._provider_factory = provider_factory
        self._predictor_factory = predictor_factory
        self._lock = threading.RLock()
        self._records: dict[str, _Record] = {}
        self._api_keys = {provider: "" for provider in sorted(KEY_PROVIDERS)}
        self._closed = False
        self._load_history()

    def set_minimax_key(self, key: str) -> bool:
        """Backward-compatible convenience wrapper for MiniMax credentials."""
        return self.set_api_key("minimax", key)

    def set_api_key(self, provider: str, key: str) -> bool:
        """Replace or clear the memory-only key used by future scan invocations."""
        if not isinstance(provider, str) or provider not in KEY_PROVIDERS:
            raise ValueError("Choose MiniMax, OpenAI, Anthropic, or Jev credentials")
        if (
            not isinstance(key, str)
            or (key and len(key) < 16)
            or len(key) > 2048
            or any(ord(char) < 33 or ord(char) > 126 for char in key)
        ):
            raise ValueError(
                "The API key must be 16–2048 printable characters without whitespace; use an empty value to clear it"
            )
        if key and any(key in name for name in SCHEMA_FIELDS):
            raise ValueError("Enter an actual provider API key, not a field name")
        with self._lock:
            if self._closed:
                raise RuntimeError("The dashboard is shutting down")
            self._api_keys[provider] = key
            return bool(key)

    def list_state(self) -> dict[str, Any]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda row: row.created_at, reverse=True)
            return {
                "credentials": {provider: bool(key) for provider, key in self._api_keys.items()},
                "features": {"vocabulary_breadth": True, "open_ended_scans": True},
                "scans": [self._snapshot(row, compact=True) for row in records],
            }

    def start(self, payload: dict[str, Any]) -> str:
        with self._lock:
            self._available()
            self._payload(
                payload,
                {
                    "target",
                    "scope",
                    "options",
                    "provider",
                    "model",
                    "budget_usd",
                    "seeds",
                    "prediction_provider",
                    "prediction_budget_usd",
                },
            )
            resolved = self.resolve(
                {key: payload[key] for key in ("target", "scope", "seeds") if key in payload}
            )
            target = resolved["target"]
            options = self._options(
                payload.get("options", {}), {"passive": resolved["passive_default"]}
            )
            seeds = resolved["seeds"]
            validate_seed_depth(target, seeds, options.max_depth)
            provider_name, model, ledger = self._provider_settings(payload)
            predictor, prediction_ledger = self._prediction_settings(payload)
            scan_id = uuid.uuid4().hex
            row = _Record(scan_id, target, asdict(options), provider_name, model, time.time())
            row.predictor = predictor
            # Resolve the user's passive/seed choices before consulting history.
            # Explicit seeds remain a focused workflow, not an implicit union.
            row.prior_verified = tuple(
                sorted(
                    {
                        name
                        for previous in self._records.values()
                        if previous.target == target
                        for name, node in previous.nodes.items()
                        if node.get("status") == "verified"
                        and name.endswith("." + target)
                        and len(name[: -(len(target) + 1)].split(".")) <= options.max_depth
                    }
                )
            )
            row.history_enabled = not seeds
            self._records[scan_id] = row
            self._launch(
                row, options, ledger, seeds, prediction_ledger=prediction_ledger, resume=False
            )
            return scan_id

    @staticmethod
    def resolve(payload: dict[str, Any]) -> dict[str, Any]:
        """Preview the same offline input resolution used to start a scan."""
        ScanManager._payload(payload, {"target", "scope", "seeds"})
        if not isinstance(payload.get("target"), str):
            raise ValueError("Enter a domain or starting subdomain")
        return asdict(
            resolve_scan_input(
                payload["target"], scope=payload.get("scope"), seeds=payload.get("seeds")
            )
        )

    def pause(self, scan_id: str) -> None:
        with self._lock:
            row = self._record(scan_id)
            if row.status not in {"running", "pausing"}:
                return
            row.cancel.set()
            row.status = "pausing"
            row.phase = "Pausing — waiting for in-flight work to finish"
            self._event(row, "phase", row.phase)

    def resume(self, scan_id: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._available()
            row = self._record(scan_id)
            if row.state is None:
                raise ValueError("No checkpoint is available for this scan")
            payload = {} if payload is None else payload
            self._payload(
                payload,
                {
                    "options",
                    "provider",
                    "model",
                    "budget_usd",
                    "seeds",
                    "prediction_provider",
                    "prediction_budget_usd",
                },
            )
            options = self._options(payload.get("options", {}), row.options)
            seeds = self._seeds(payload.get("seeds", []), row.target)
            validate_seed_depth(row.target, seeds, options.max_depth)
            provider_name, model, ledger = self._provider_settings(payload, row.state)
            predictor, prediction_ledger = self._prediction_settings(payload, row.state)
            row.options = asdict(options)
            row.provider, row.model = provider_name, model
            row.predictor = predictor
            row.status, row.phase, row.error = "running", "Resuming saved scan", None
            row.cancel = threading.Event()
            row.active.clear()
            row.snapshot_at = time.monotonic()
            self._launch(
                row, options, ledger, seeds, prediction_ledger=prediction_ledger, resume=True
            )

    def snapshot(self, scan_id: str) -> dict[str, Any]:
        with self._lock:
            return self._snapshot(self._record(scan_id))

    def report(self, scan_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._record(scan_id)
            snapshot = self._snapshot(row)
            prior = set(row.prior_verified)
            current = set(snapshot["summary"]["findings"])
            return {
                **snapshot["summary"],
                "id": row.id,
                "target": row.target,
                "status": row.status,
                "patterns": snapshot["patterns"],
                "nodes": snapshot["nodes"],
                "options": copy.deepcopy(row.options),
                "provider": row.provider,
                "model": row.model,
                "comparison": {
                    "prior_verified": list(row.prior_verified),
                    "retained": sorted(prior & current),
                    "added": sorted(current - prior),
                    # Includes pending, budget-limited and inconclusive names:
                    # lack of current verification does not establish absence.
                    "not_reverified": sorted(prior - current),
                },
            }

    def close(self) -> None:
        """Request pauses without blocking HTTP shutdown on provider timeouts."""
        with self._lock:
            self._closed = True
            self._api_keys = {provider: "" for provider in sorted(KEY_PROVIDERS)}
            for row in self._records.values():
                if row.status in {"running", "pausing"}:
                    self.pause(row.id)

    def _available(self) -> None:
        if self._closed:
            raise RuntimeError("The dashboard is shutting down")
        if any(row.status in {"running", "pausing"} for row in self._records.values()):
            raise RuntimeError("Pause the active scan before starting or resuming another")

    def _record(self, scan_id: str) -> _Record:
        if (
            not isinstance(scan_id, str)
            or not SCAN_ID.fullmatch(scan_id)
            or scan_id not in self._records
        ):
            raise KeyError("Unknown scan")
        return self._records[scan_id]

    @staticmethod
    def _payload(payload: Any, allowed: set[str]) -> None:
        if not isinstance(payload, dict) or any(key not in allowed for key in payload):
            raise ValueError("Scan request contains unsupported fields")

    @staticmethod
    def _options(raw: Any, previous: dict[str, Any] | None = None) -> ScanOptions:
        if not isinstance(raw, dict) or any(
            name not in ScanOptions.__dataclass_fields__ for name in raw
        ):
            raise ValueError("Scan options contain unsupported fields")
        return ScanOptions(**{**(previous or {}), **raw})

    @staticmethod
    def _seeds(raw: Any, target: str) -> list[str]:
        if (
            not isinstance(raw, list)
            or len(raw) > 1000
            or any(not isinstance(item, str) or len(item) > 253 for item in raw)
        ):
            raise ValueError("Seeds must be a list of at most 1000 subdomain names")
        values = []
        for item in raw:
            value = normalize_target(item)
            if not value.endswith("." + target):
                raise ValueError("Every seed must be a subdomain of the target")
            if value not in values:
                values.append(value)
        return values

    def _provider_settings(
        self, payload: dict[str, Any], state: dict[str, Any] | None = None
    ) -> tuple[str, str | None, SpendLedger | None]:
        saved_provider = state.get("provider") if state else None
        default_provider = next(
            (
                name
                for name, identity in IDENTITIES.items()
                if saved_provider and saved_provider["name"] == identity
            ),
            "none",
        )
        provider = payload.get("provider", default_provider)
        if not isinstance(provider, str) or provider not in {"none", *MODELS}:
            raise ValueError("Choose free discovery, MiniMax, OpenAI, or Anthropic")
        saved_model = saved_provider["model"] if saved_provider else None
        model = payload.get(
            "model", saved_model or (MODELS[provider][0] if provider != "none" else None)
        )
        if provider == "none" and model is not None:
            raise ValueError("A model can only be selected for hosted AI scans")
        if provider != "none" and (not isinstance(model, str) or model not in MODELS[provider]):
            raise ValueError("Choose a supported model for the selected provider")
        if saved_provider:
            if provider == "none" or saved_provider != {
                "name": IDENTITIES[provider],
                "model": model,
            }:
                raise ValueError("Resume requires the same provider and model as the saved scan")
        elif state is not None and provider != "none":
            raise ValueError("Start a new scan to enable hosted AI on a free scan")
        budget = payload.get("budget_usd", None)
        if budget is not None and (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(budget)
            or budget < 0
        ):
            raise ValueError("AI budget must be a finite nonnegative number in USD")
        if provider == "none":
            if budget is not None:
                raise ValueError("A dollar budget is only needed for hosted AI scans")
            return provider, None, None
        if not self._api_keys[provider]:
            raise ValueError(
                "Add an API key for the selected provider in the dashboard before starting an AI scan"
            )
        if state is not None and state.get("ledger") is not None:
            ledger = SpendLedger.from_dict(state["ledger"])
            if budget is not None:
                if Decimal(str(budget)) < Decimal(ledger.to_dict()["spent_usd"]):
                    raise ValueError(
                        "The cumulative budget cannot be below previously spent or reserved charges"
                    )
                restored = ledger.to_dict()
                restored["limit_usd"] = str(budget)
                ledger = SpendLedger.from_dict(restored)
        else:
            ledger = SpendLedger(1 if budget is None else budget, price_for(provider, str(model)))
        return provider, str(model), ledger

    def _prediction_settings(
        self, payload: dict[str, Any], state: dict[str, Any] | None = None
    ) -> tuple[dict[str, str] | None, SpendLedger | None]:
        saved = state.get("predictor") if state else None
        name = payload.get("prediction_provider", "jev" if saved else "none")
        if not isinstance(name, str) or name not in {"none", "jev"}:
            raise ValueError("Choose no prediction provider or Jev")
        if state is not None and name != ("jev" if saved else "none"):
            raise ValueError(
                "Resume requires the same prediction provider; start a new scan to change it"
            )
        budget = payload.get("prediction_budget_usd")
        if budget is not None and (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(budget)
            or budget < 0
        ):
            raise ValueError(
                "The additional prediction budget must be a finite nonnegative number in USD"
            )
        if name == "none":
            if budget is not None:
                raise ValueError("Select Jev to set an additional prediction budget")
            return None, None
        from ..providers.jev import JEV_MODEL, JEV_PRICE

        identity = {"name": "jev", "model": JEV_MODEL}
        if saved is not None and saved != identity:
            raise ValueError("Resume requires the saved Jev model")
        if not self._api_keys["jev"]:
            raise ValueError("Add your Jev / TypeSafe API key before enabling prediction")
        if state is not None and state.get("prediction_ledger") is not None:
            ledger = SpendLedger.from_dict(state["prediction_ledger"])
            if budget is not None:
                if Decimal(str(budget)) < Decimal(ledger.to_dict()["spent_usd"]):
                    raise ValueError(
                        "The cumulative prediction budget cannot be below spent or reserved charges"
                    )
                restored = ledger.to_dict()
                restored["limit_usd"] = str(budget)
                ledger = SpendLedger.from_dict(restored)
        elif state is not None:
            raise ValueError("Saved Jev scan is missing its prediction budget")
        else:
            ledger = SpendLedger(0.05 if budget is None else budget, JEV_PRICE)
        return identity, ledger

    def _launch(
        self,
        row: _Record,
        options: ScanOptions,
        ledger: SpendLedger | None,
        seeds: list[str],
        *,
        prediction_ledger: SpendLedger | None = None,
        resume: bool,
    ) -> None:
        key = self._api_keys.get(row.provider, "")
        prediction_key = self._api_keys["jev"] if row.predictor is not None else ""
        self._event(row, "phase", row.phase)
        thread = threading.Thread(
            target=self._work,
            args=(row.id, options, ledger, seeds, key, prediction_key, prediction_ledger, resume),
            name=f"enumeraite-ui-{row.id[:8]}",
            daemon=True,
        )
        row.thread = thread
        thread.start()

    def _work(
        self,
        scan_id: str,
        options: ScanOptions,
        ledger: SpendLedger | None,
        seeds: list[str],
        key: str,
        prediction_key: str,
        prediction_ledger: SpendLedger | None,
        resume: bool,
    ) -> None:
        with self._lock:
            row = self._records[scan_id]
            saved_state = copy.deepcopy(row.state)
        store = ScanStore(self.storage_dir / f"{scan_id}.json")
        provider = None
        predictor = None
        run_keys = (key, prediction_key)
        try:
            with store.locked(resume=resume):
                if resume:
                    latest = store.load()
                    validate_checkpoint(latest, row.target)
                    latest = _state_projection(latest)
                    expected = _state_projection(saved_state) if saved_state else None
                    if latest != expected:
                        # Another dashboard may have advanced this ledger while
                        # this window was idle. Never spend or overwrite using
                        # stale totals, even after the advisory lock is released.
                        self._adopt_changed_checkpoint(row, latest)
                        return
                if row.provider != "none":
                    factory = self._provider_factory or (
                        AnthropicProvider
                        if row.provider == "anthropic"
                        else OpenAICompatibleProvider
                    )
                    provider = factory(
                        {
                            "preset": row.provider,
                            "model": row.model,
                            "api_key": key,
                            "base_url": ENDPOINTS[row.provider],
                            "max_retries": 0,
                            "timeout": 30,
                            "hybrid": False,
                            "batch_size": 20,
                            "max_batch_calls": 1,
                        }
                    )
                prediction_args: dict[str, Any] = {}
                if row.predictor is not None:
                    from ..providers.jev import JevPredictor

                    predictor = (self._predictor_factory or JevPredictor)(
                        api_key=prediction_key, model=row.predictor["model"]
                    )
                    prediction_args = {
                        "predictor": predictor,
                        "prediction_ledger": prediction_ledger,
                    }
                validator = self._dns_factory(
                    row.target,
                    workers=options.workers,
                    rate=options.rate,
                    timeout=options.dns_timeout,
                    cancel_event=row.cancel,
                    event=lambda kind, data: self._on_event(row, kind, data, run_keys),
                )
                session = self._session_factory(
                    row.target,
                    options,
                    validator,
                    passive_fetch=self._passive_fetch,
                    provider=provider,
                    ledger=ledger,
                    state=saved_state,
                    checkpoint=lambda state: self._checkpoint(row, store, state, run_keys),
                    event=lambda kind, data: self._on_event(row, kind, data, run_keys),
                    stop_requested=row.cancel.is_set,
                    **prediction_args,
                )
                # Establish a resumable checkpoint even if paused during passive work.
                session.save()
                if row.history_enabled and row.prior_verified:
                    session.run(seeds, prior_names=row.prior_verified)
                else:
                    session.run(seeds)
                final_status = "paused" if row.cancel.is_set() else "completed"
                self._checkpoint(row, store, session.state, run_keys, final_status=final_status)
                with self._lock:
                    row.status = final_status
                    row.phase = (
                        "Paused — progress saved" if row.status == "paused" else "Scan complete"
                    )
                    row.active.clear()
                    self._event(row, "phase", row.phase)
        except Exception:
            # Provider exceptions can contain arbitrary headers, response bodies,
            # keys or local paths. Do not expose their text to the UI or disk.
            with self._lock:
                row.status = "error"
                row.error = "The scan worker stopped unexpectedly. Saved progress can be resumed."
                row.phase = "Scan stopped"
                row.active.clear()
                self._event(row, "error", row.error)
        finally:
            if predictor is not None:
                try:
                    predictor.close()
                except Exception:
                    with self._lock:
                        self._event(row, "warning", "Prediction connection cleanup failed")
            if provider is not None:
                try:
                    client = getattr(provider, "client", None)
                    if client is not None:
                        client.close()
                except Exception:
                    with self._lock:
                        self._event(row, "warning", "Provider connection cleanup failed")

    def _redact(self, value: Any, run_key: str | tuple[str, ...] = "") -> Any:
        supplied = (run_key,) if isinstance(run_key, str) else run_key
        keys = {key for key in (*supplied, *self._api_keys.values()) if key}
        if isinstance(value, str):
            for key in keys:
                value = value.replace(key, "[REDACTED]")
            return safe_model_text(value, 4000)
        if isinstance(value, dict):
            return {
                k if k in SCHEMA_FIELDS else self._redact(k, run_key): self._redact(v, run_key)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._redact(item, run_key) for item in value]
        return value

    def _adopt_changed_checkpoint(self, row: _Record, state: dict[str, Any]) -> None:
        """Reload another process's progress, without writing or making a request."""
        identity = state.get("provider")
        provider = next(
            (name for name, value in IDENTITIES.items() if identity and value == identity["name"]),
            "none",
        )
        if identity and (provider == "none" or identity["model"] not in MODELS[provider]):
            raise ValueError("Saved scan has an unsupported provider")
        clean = self._redact(copy.deepcopy(state))
        nodes = {node["value"]: node for node in clean["nodes"]}
        findings = tuple(value for value, node in nodes.items() if node["status"] == "verified")
        patterns = describe_patterns(findings, row.target)
        with self._lock:
            row.state = clean
            row.nodes = nodes
            row.options = clean["options"]
            row.provider = provider
            row.model = identity["model"] if identity else None
            row.predictor = clean.get("predictor")
            row.patterns = patterns
            row.pattern_findings = findings
            row.elapsed = clean["elapsed_seconds"]
            row.snapshot_at = time.monotonic()
            row.status = "error"
            row.error = (
                "This scan was updated in another dashboard. Latest progress loaded; "
                "review its limits and resume again."
            )
            row.phase = "Updated progress loaded"
            row.active.clear()
            self._event(row, "warning", row.error)

    def _checkpoint(
        self,
        row: _Record,
        store: ScanStore,
        state: dict[str, Any],
        key: str | tuple[str, ...],
        *,
        final_status: str | None = None,
    ) -> None:
        clean = self._redact(
            _state_projection(state),
            key,
        )
        with self._lock:
            clean["_ui"] = self._redact(
                {
                    "created_at": row.created_at,
                    "status": final_status or row.status,
                    "prior_verified": list(row.prior_verified),
                    "history_enabled": row.history_enabled,
                },
                key,
            )
        store.save(clean)
        nodes = {node["value"]: node for node in clean["nodes"]}
        with self._lock:
            old_findings = row.pattern_findings
        findings = tuple(value for value, node in nodes.items() if node["status"] == "verified")
        patterns = describe_patterns(findings, row.target) if findings != old_findings else None
        with self._lock:
            row.state = clean
            row.nodes = nodes
            if patterns is not None:
                row.patterns = patterns
                row.pattern_findings = findings
            row.elapsed = clean["elapsed_seconds"]
            row.snapshot_at = time.monotonic()
            row.revision += 1

    def _event(self, row: _Record, kind: str, message: str, hostname: str | None = None) -> None:
        row.event_id += 1
        entry = {"id": row.event_id, "time": time.time(), "kind": kind, "message": message}
        if hostname:
            entry["hostname"] = hostname
        row.events.append(entry)
        row.revision += 1

    def _on_event(
        self, row: _Record, kind: str, payload: dict[str, Any], key: str | tuple[str, ...]
    ) -> None:
        with self._lock:
            data = self._redact(payload, key)
            hostname = data.get("hostname", data.get("value"))
            if kind == "dns_start":
                query_id = str(data.get("query_id", ""))
                if len(row.active) < 200:
                    row.active[query_id] = {
                        "id": query_id,
                        "hostname": hostname,
                        "candidate": data.get("candidate", hostname),
                        "query_type": data.get("query_kind", "DNS"),
                        "worker_id": data.get("worker_id"),
                        "probe": bool(data.get("probe")),
                        "started_at": time.time(),
                    }
                row.revision += 1
                return
            if kind == "dns_end":
                row.active.pop(str(data.get("query_id", "")), None)
                row.revision += 1
                return
            if kind in {"candidate", "result", "finding"} and isinstance(data.get("value"), str):
                value = data["value"]
                if value.endswith("." + row.target) or value in row.nodes:
                    row.nodes[value] = {**row.nodes.get(value, {}), **data}
            if kind == "phase" and row.status != "pausing":
                row.phase = str(data.get("message", "Working"))[:1000]
            messages = {
                "candidate": f"Queued {hostname}",
                "finding": f"Verified {hostname}",
                "result": f"{hostname}: {data.get('status', 'checked')}",
                "passive": f"Collected {data.get('count', 0)} known subdomains",
                "progress": f"Verified {data.get('verified', 0)} subdomains",
            }
            message = str(data.get("message", messages.get(kind, kind)))[:1000]
            self._event(row, kind, message, hostname if isinstance(hostname, str) else None)

    def _summary(self, row: _Record) -> dict[str, Any]:
        state = row.state or {}
        nodes = list(row.nodes.values())
        elapsed = row.elapsed
        if row.status in {"running", "pausing"}:
            elapsed += max(0, time.monotonic() - row.snapshot_at)
        findings = [node["value"] for node in nodes if node.get("status") == "verified"]
        return {
            "target": row.target,
            "findings": findings,
            "verified": len(findings),
            "known_verified": sum(
                node.get("status") == "verified"
                and node.get("source") in {"seed", "passive", "history"}
                for node in nodes
            ),
            "attempted": sum(node.get("attempts", 0) > 0 for node in nodes),
            "queued": sum(node.get("status") == "pending" for node in nodes),
            "inconclusive": sum(node.get("status") == "inconclusive" for node in nodes),
            "wildcard": sum(node.get("status") == "wildcard" for node in nodes),
            "round": state.get("round", 0),
            "discovery_methods": discovery_method_counts(nodes),
            "elapsed_seconds": elapsed,
            "ledger": state.get("ledger"),
            "predictor": state.get("predictor", row.predictor),
            "prediction_ledger": state.get("prediction_ledger"),
            "predictions": state.get("predictions", []),
            "warnings": state.get("warnings", []),
            "stop_reason": state.get("stop_reason", "starting"),
            "passive_sources": state.get("passive_sources", {}),
            "passive_truncated": state.get("passive_truncated", False),
        }

    def _snapshot(self, row: _Record, *, compact: bool = False) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "id": row.id,
            "target": row.target,
            "status": row.status,
            "phase": row.phase,
            "options": row.options,
            "provider": row.provider,
            "model": row.model,
            "summary": self._summary(row),
            "created_at": row.created_at,
            "error": row.error,
            "can_resume": row.status not in {"running", "pausing"} and row.state is not None,
            "revision": row.revision,
        }
        if not compact:
            snapshot.update(
                nodes=list(row.nodes.values()),
                patterns=row.patterns,
                active=list(row.active.values()),
                events=list(row.events),
            )
        else:
            snapshot["summary"].pop("findings", None)
            snapshot["summary"].pop("passive_sources", None)
        return copy.deepcopy(snapshot)

    def _load_history(self) -> None:
        if not self.storage_dir.exists():
            return
        # Load every saved session without deleting history or following supplied
        # paths and symlinks. ScanStore also rejects non-regular files.
        for path in self.storage_dir.iterdir():
            if path.suffix != ".json" or not SCAN_ID.fullmatch(path.stem) or path.is_symlink():
                continue
            try:
                state = ScanStore(path).load()
                if not isinstance(state.get("target"), str):
                    continue
                target = normalize_target(state["target"])
                validate_checkpoint(state, target)
                provider = state.get("provider")
                provider_name = next(
                    (
                        name
                        for name, identity in IDENTITIES.items()
                        if provider and provider["name"] == identity
                    ),
                    "none",
                )
                if provider and (
                    provider_name == "none" or provider["model"] not in MODELS[provider_name]
                ):
                    continue
                metadata = state.get("_ui", {})
                if not isinstance(metadata, dict):
                    continue
                prior_verified = metadata.get("prior_verified", [])
                if not isinstance(prior_verified, list) or any(
                    not isinstance(name, str)
                    or normalize_target(name) != name
                    or not name.endswith("." + target)
                    for name in prior_verified
                ):
                    continue
                history_enabled = metadata.get("history_enabled", False)
                if not isinstance(history_enabled, bool):
                    continue
                created = metadata.get("created_at", path.stat().st_mtime)
                if (
                    isinstance(created, bool)
                    or not isinstance(created, (int, float))
                    or not math.isfinite(created)
                ):
                    continue
                status = "completed" if metadata.get("status") == "completed" else "paused"
                row = _Record(
                    path.stem,
                    target,
                    state["options"],
                    provider_name,
                    provider["model"] if provider else None,
                    created,
                    status=status,
                    phase="Saved scan — ready to resume" if status == "paused" else "Scan complete",
                    state=_state_projection(state),
                    nodes={node["value"]: node for node in state["nodes"]},
                    elapsed=state["elapsed_seconds"],
                    predictor=state.get("predictor"),
                    prior_verified=tuple(sorted(set(prior_verified))),
                    history_enabled=history_enabled,
                )
                row.patterns = describe_patterns(
                    [node["value"] for node in state["nodes"] if node["status"] == "verified"],
                    target,
                )
                row.pattern_findings = tuple(
                    node["value"] for node in state["nodes"] if node["status"] == "verified"
                )
                self._records[row.id] = row
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                # Invalid files are ignored, not echoed into browser error logs.
                continue
