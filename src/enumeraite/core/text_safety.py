"""Small output-safety helpers for text returned by remote models."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])")
_SECRET_TOKEN = re.compile(r"(?i)(?:sk-(?:api-|proj-|ant-)?)[a-z0-9_\\-]{12,}")
_PREFIXED_TOKEN = re.compile(r"(?i)\b(?:gsk_|hf_)[a-z0-9_\\-]{12,}")
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+\\/-]{12,}")
_LABELED_SECRET = re.compile(
    r"(?i)(\b(?:api[_ -]?key|authorization|x-api-key)\b[\"']?\s*[:=]\s*[\"']?)"
    r"([^\"'\s,;}\]]{8,})"
)


def safe_model_text(value: Any, limit: int, multiline: bool = False) -> str:
    """Strip terminal controls and bound model-authored display text."""
    text = _ANSI_ESCAPE.sub("", str(value or ""))
    if not multiline:
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    allowed_controls = {"\n", "\t"} if multiline else set()
    text = "".join(
        char
        for char in text
        if char in allowed_controls or not unicodedata.category(char).startswith("C")
    )
    if not multiline:
        text = " ".join(text.split())
    return text.strip()[: max(0, limit)]


def safe_exception_text(value: Any, limit: int = 2_000) -> str:
    """Sanitize exception text and redact recognizable credential tokens."""
    text = safe_model_text(value, max(limit * 2, limit))
    text = _SECRET_TOKEN.sub("<redacted-api-key>", text)
    text = _PREFIXED_TOKEN.sub("<redacted-api-key>", text)
    text = _BEARER_TOKEN.sub("Bearer <redacted-token>", text)
    text = _LABELED_SECRET.sub(r"\1<redacted-token>", text)
    return text[: max(0, limit)]


def json_safe_value(value: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    """Return a bounded, acyclic value accepted by strict JSON encoders."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= 16_384 else 0
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, str):
        return safe_model_text(value, 100_000, multiline=True)
    if _depth >= 10:
        return safe_model_text(value, 500)

    seen = _seen if _seen is not None else set()
    if isinstance(value, dict):
        identity = id(value)
        if identity in seen:
            return "<cycle>"
        seen.add(identity)
        mapping_result = {
            safe_model_text(key, 200): json_safe_value(item, _depth=_depth + 1, _seen=seen)
            for index, (key, item) in enumerate(value.items())
            if index < 10_000
        }
        seen.remove(identity)
        return mapping_result
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return ["<cycle>"]
        seen.add(identity)
        sequence_result = [
            json_safe_value(item, _depth=_depth + 1, _seen=seen) for item in value[:10_000]
        ]
        seen.remove(identity)
        return sequence_result
    return safe_model_text(value, 500)
