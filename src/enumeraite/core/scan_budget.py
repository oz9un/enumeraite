"""Conservative, resumable model spending controls for interactive scans.

Amounts are estimates of token charges, not provider invoices. Reservations are
made before every request, include the entire completion allowance, and survive
an interrupted process. The registry uses conservative published rates; custom
endpoints and unknown models require explicit user-supplied prices.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..providers.llm_provider import ChatResult, LLMProvider


class BudgetExhausted(RuntimeError):
    """No further model request can be safely admitted by this scan."""


class ScanPaused(RuntimeError):
    """A cooperative pause prevents admitting a new paid request."""


def _amount(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, with provenance persisted with the session."""

    input_usd_per_million: float
    output_usd_per_million: float
    cached_input_usd_per_million: float | None = None
    source: str = "user override"
    checked_at: str = ""

    def __post_init__(self) -> None:
        _amount(self.input_usd_per_million, "input price")
        _amount(self.output_usd_per_million, "output price")
        if self.cached_input_usd_per_million is not None:
            _amount(self.cached_input_usd_per_million, "cache price")
            if self.cached_input_usd_per_million > self.input_usd_per_million:
                raise ValueError("cache price must not exceed the conservative input price")
        if not isinstance(self.source, str) or not isinstance(self.checked_at, str):
            raise ValueError("price provenance must be text")


_MINIMAX_SOURCE = "https://platform.minimax.io/docs/pricing/overview"
_OPENAI_MINI_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4.1-mini"
_OPENAI_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4.1"
_ANTHROPIC_SOURCE = "https://platform.claude.com/docs/en/models/sonnet-4-6/overview"
_CHECKED_AT = "2026-10-02"
_PRICES = {
    # M3 uses the higher >512k standard context tier even for shorter prompts.
    # M2.7 uses the cache-write input ceiling, avoiding under-reserving when
    # automatic caching creates entries. Standard service only, no server tools.
    ("minimax", "MiniMax-M3"): ModelPrice(0.60, 2.40, 0.12, _MINIMAX_SOURCE, _CHECKED_AT),
    ("minimax", "MiniMax-M2.7"): ModelPrice(0.375, 1.20, 0.06, _MINIMAX_SOURCE, _CHECKED_AT),
    ("minimax", "MiniMax-M2.7-highspeed"): ModelPrice(
        0.60, 2.40, 0.06, _MINIMAX_SOURCE, _CHECKED_AT
    ),
    ("openai", "gpt-4.1-mini"): ModelPrice(0.40, 1.60, 0.10, _OPENAI_MINI_SOURCE, _CHECKED_AT),
    ("openai", "gpt-4.1"): ModelPrice(2.0, 8.0, 0.50, _OPENAI_SOURCE, _CHECKED_AT),
    # Use the one-hour cache-write ceiling for conservative input reservations.
    ("anthropic", "claude-sonnet-4-6"): ModelPrice(6.0, 15.0, 0.30, _ANTHROPIC_SOURCE, _CHECKED_AT),
}
_LOCAL_PRESETS = {"local", "ollama", "vllm", "llamacpp"}


def price_for(
    provider_name: str,
    model: str,
    input_price: float | None = None,
    output_price: float | None = None,
) -> ModelPrice:
    """Resolve conservative rates; overrides are USD per million tokens."""
    if input_price is not None or output_price is not None:
        if input_price is None or output_price is None:
            raise ValueError("supply both input and output prices in USD per million tokens")
        return ModelPrice(input_price, output_price)
    preset = provider_name.removeprefix("openai-compat:").lower()
    if preset in _LOCAL_PRESETS:
        return ModelPrice(0.0, 0.0, source="local loopback service; infrastructure excluded")
    price = _PRICES.get((preset, model))
    if price is None:
        raise ValueError(
            f"no verified scan price for {provider_name}/{model}; supply both "
            "--input-price and --output-price (USD per million tokens)"
        )
    return price


@dataclass(frozen=True)
class Reservation:
    input_tokens: int
    max_output_tokens: int
    amount: Decimal


class SpendLedger:
    """A single serial request ledger with durable conservative reservations."""

    def __init__(
        self,
        limit_usd: float | Decimal,
        pricing: ModelPrice,
        *,
        spent_usd: float = 0.0,
        reserved_usd: float = 0.0,
        calls: int = 0,
        uncertain_calls: int = 0,
        blocked: bool = False,
    ):
        self._limit = _amount(limit_usd, "budget")
        self._spent = _amount(spent_usd, "spent amount")
        self._reserved = _amount(reserved_usd, "reserved amount")
        if not isinstance(pricing, ModelPrice):
            raise ValueError("pricing must be a ModelPrice")
        self.pricing = pricing
        self.calls = _count(calls, "request count")
        self.uncertain_calls = _count(uncertain_calls, "uncertain request count")
        if not isinstance(blocked, bool):
            raise ValueError("blocked must be a boolean")
        self.blocked = blocked

    @property
    def limit_usd(self) -> float:
        return float(self._limit)

    @property
    def spent_usd(self) -> float:
        return float(self._spent)

    @property
    def reserved_usd(self) -> float:
        return float(self._reserved)

    @property
    def remaining_usd(self) -> float:
        return float(max(Decimal(0), self._limit - self._spent - self._reserved))

    def reserve(self, input_tokens: int, max_output_tokens: int) -> Reservation:
        """Reserve a whole request, reducing its output cap if necessary."""
        _count(input_tokens, "input token estimate")
        _count(max_output_tokens, "output token limit")
        if max_output_tokens < 1:
            raise ValueError("output token limit must be positive")
        if self.blocked:
            raise BudgetExhausted("model spending stopped after inconsistent usage reporting")
        if self._reserved:
            raise BudgetExhausted("a previous model request still has a reserved charge")
        available = max(Decimal(0), self._limit - self._spent)
        input_cost = self._cost(input_tokens, 0)
        if input_cost > available:
            raise BudgetExhausted("remaining model budget cannot cover this prompt")
        output_rate = Decimal(str(self.pricing.output_usd_per_million)) / 1_000_000
        allowed = max_output_tokens
        if output_rate:
            allowed = min(
                allowed,
                int(((available - input_cost) / output_rate).to_integral_value(ROUND_FLOOR)),
            )
        if allowed < 1:
            raise BudgetExhausted("remaining model budget cannot cover a completion")
        amount = self._cost(input_tokens, allowed)
        self._reserved = amount
        self.calls += 1
        return Reservation(input_tokens, allowed, amount)

    def settle(self, reservation: Reservation, response: ChatResult | None) -> None:
        """Charge usage or the entire reservation if the outcome is uncertain."""
        self._reserved = Decimal(0)
        if (
            response is None
            or not response.usage_complete
            or response.input_tokens < 1
            or (response.text and response.output_tokens == 0)
            or response.output_tokens < response.reasoning_tokens
        ):
            self._spent += reservation.amount
            self.uncertain_calls += 1
            return
        charge = self._cost(
            response.input_tokens, response.output_tokens, response.cached_input_tokens
        )
        self._spent += charge
        # A provider can violate its output limit or a tokenizer can exceed the
        # byte estimate. Record the real estimate (never hide an overshoot) and
        # refuse further requests, even if cache discounts made this one cheap.
        if (
            response.input_tokens > reservation.input_tokens
            or response.output_tokens > reservation.max_output_tokens
            or charge > reservation.amount
            or self._spent > self._limit
        ):
            self.blocked = True

    def cancel(self) -> None:
        """Release a reservation only when no network request was started."""
        self._reserved = Decimal(0)
        self.calls = max(0, self.calls - 1)

    def _cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> Decimal:
        price = self.pricing
        # Unknown cached-token pricing receives no speculative discount.
        cached_rate = (
            price.input_usd_per_million
            if price.cached_input_usd_per_million is None
            else price.cached_input_usd_per_million
        )
        cached = min(input_tokens, max(0, cached_tokens))
        return (
            (input_tokens - cached) * Decimal(str(price.input_usd_per_million))
            + cached * Decimal(str(cached_rate))
            + output_tokens * Decimal(str(price.output_usd_per_million))
        ) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        """Serialize exact amounts without credentials or prompt content."""
        return {
            "limit_usd": str(self._limit),
            "spent_usd": str(self._spent),
            "reserved_usd": str(self._reserved),
            "pricing": asdict(self.pricing),
            "calls": self.calls,
            "uncertain_calls": self.uncertain_calls,
            "blocked": self.blocked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpendLedger:
        """Treat interrupted requests as fully spent, never as reusable credit."""
        if not isinstance(data, dict) or not isinstance(data.get("pricing"), dict):
            raise ValueError("invalid saved model budget")
        try:
            price = ModelPrice(**data["pricing"])
        except TypeError as exc:
            raise ValueError("invalid saved model pricing") from exc
        ledger = cls(
            _saved_amount(data.get("limit_usd"), "budget"),
            price,
            calls=data.get("calls", 0),
            uncertain_calls=data.get("uncertain_calls", 0),
            blocked=data.get("blocked", False),
        )
        spent = _saved_amount(data.get("spent_usd", "0"), "spent amount")
        reserved = _saved_amount(data.get("reserved_usd", "0"), "reserved amount")
        ledger._spent = spent + reserved
        if reserved:
            ledger.uncertain_calls += 1
        return ledger


def attach_scan_guard(
    provider: LLMProvider,
    ledger: SpendLedger,
    deadline: float | None,
    on_change: Callable[[], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Protect every chat call, including calls made inside a generation batch.

    The synchronous SDK receives the remaining timeout and has retries disabled.
    Its timeout is an I/O timeout, not an interruptible whole-process deadline;
    an in-flight request may take longer to unwind. No later request is admitted
    once the monotonic scan deadline expires.
    """
    if getattr(provider, "_scan_guard_attached", False):
        raise ValueError("a scan budget guard is already attached to this provider")
    if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic time")
    _validate_price_endpoint(provider, ledger.pricing)
    original_chat = provider.chat
    provider._scan_guard_attached = True

    def guarded_chat(system: str, user: str, schema: dict[str, Any] | None) -> ChatResult:
        if stop_requested and stop_requested():
            raise ScanPaused("Scan paused before the model request")
        timeout = _remaining_timeout(provider.request_timeout, deadline)
        # UTF-8 bytes are a deliberately large text token estimate. Include
        # schema serialization and a 1024-token allowance for tool/chat framing.
        input_bound = (
            len(json.dumps([system, user, schema], ensure_ascii=True).encode("utf-8")) + 1024
        )
        reservation = ledger.reserve(input_bound, provider.max_tokens)
        reservation_saved = False
        try:
            if on_change:
                on_change()  # Durable reservation MUST precede the network call.
                reservation_saved = True
            if stop_requested and stop_requested():
                raise ScanPaused("Scan paused before the model request")
            timeout = min(timeout, _remaining_timeout(provider.request_timeout, deadline))
        except BaseException:
            ledger.cancel()
            if reservation_saved and on_change:
                on_change()  # No request was sent: persist the released reservation.
            raise
        saved_max_tokens = provider.max_tokens
        saved_timeout = provider.request_timeout
        provider.max_tokens = reservation.max_output_tokens
        provider.request_timeout = timeout
        try:
            response = original_chat(system, user, schema)
        except BaseException:
            ledger.settle(reservation, None)
            if on_change:
                on_change()
            raise
        else:
            ledger.settle(reservation, response)
            if on_change:
                on_change()
            if ledger.blocked:
                raise BudgetExhausted("model usage exceeded its reservation")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("scan time limit reached during the model request")
            return response
        finally:
            provider.max_tokens = saved_max_tokens
            provider.request_timeout = saved_timeout

    provider.chat = guarded_chat  # type: ignore[method-assign]


def _validate_price_endpoint(provider: LLMProvider, price: ModelPrice) -> None:
    """Registry prices apply only to the endpoint they describe."""
    from .url_safety import is_loopback_host, parse_http_url

    if price.source == "user override":
        return
    endpoint = str(provider.config.get("base_url", "")).rstrip("/")
    if not endpoint:
        endpoint = str(getattr(getattr(provider, "client", None), "base_url", "")).rstrip("/")
    if price.source == _MINIMAX_SOURCE and endpoint != "https://api.minimax.io/v1":
        raise ValueError("custom MiniMax endpoints require explicit input and output prices")
    if (
        price.source in {_OPENAI_SOURCE, _OPENAI_MINI_SOURCE}
        and endpoint != "https://api.openai.com/v1"
    ):
        raise ValueError("custom OpenAI endpoints require explicit input and output prices")
    if price.source == _ANTHROPIC_SOURCE and endpoint != "https://api.anthropic.com":
        raise ValueError("custom Anthropic endpoints require explicit input and output prices")
    if price.source.startswith("local loopback"):
        parsed = parse_http_url(endpoint, label="local model endpoint")
        if not parsed.hostname or not is_loopback_host(parsed.hostname):
            raise ValueError("remote local-model endpoints require explicit token prices")


def _remaining_timeout(configured: float, deadline: float | None) -> float:
    if deadline is None:
        return configured
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("scan time limit reached before the model request")
    return min(configured, remaining)


def _saved_amount(value: Any, name: str) -> Decimal:
    if isinstance(value, str):
        try:
            value = Decimal(value)
        except ArithmeticError as exc:
            raise ValueError(f"invalid saved {name}") from exc
    return _amount(value, name)


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value
