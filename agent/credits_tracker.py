"""Credits tracking for Nous inference API responses.

Captures x-nous-credits-* headers from streaming responses and exposes a
session-cumulative spend figure for a dev-only TUI readout.  This is the L0
data-path slice of the usage-aware-credits feature: header -> state -> TUI.

Header schema (x-nous-credits-* family):
    x-nous-credits-version                    contract/schema version
    x-nous-credits-remaining-micros           total remaining balance (micros)
    x-nous-credits-remaining-usd              same, formatted USD string
    x-nous-credits-subscription-micros        subscription balance (SIGNED; may be negative/debt)
    x-nous-credits-subscription-usd           same, formatted USD string
    x-nous-credits-subscription-limit-micros  subscription cap (PAIRED/optional)
    x-nous-credits-subscription-limit-usd     same, formatted USD string (PAIRED/optional)
    x-nous-credits-rollover-micros            rolled-over balance (micros)
    x-nous-credits-purchased-micros           purchased balance (micros)
    x-nous-credits-purchased-usd              same, formatted USD string
    x-nous-credits-denominator-kind           "subscription_cap" | "none"
    x-nous-credits-paid-access                "true" | "false" (STRING!)
    x-nous-credits-disabled-reason            reason string (header omitted when null)
    x-nous-credits-as-of-ms                   server-side timestamp (ms epoch)

Money is handled as micros ints only; *_usd values are preserved verbatim as
the raw strings the server sent (never re-parsed to float).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional


def _safe_int(value: Any, default: int = 0) -> int:
    """Parse a header value to an exact int (money-safe).

    The contract guarantees every ``*_micros`` field is an integer string, so
    we parse with ``int()`` directly — NOT ``int(float(...))`` — to avoid the
    float-precision loss above 2**53 that would silently corrupt large money
    values.  A non-integer or out-of-range value falls back to ``default``
    rather than raising (fail-open: a single malformed header must never break
    the agent loop).  ``OverflowError`` is caught explicitly because the float
    fallback below can raise it on ``"inf"`` / ``"1e400"`` inputs.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    # Tolerate float-shaped strings ("1.0") by truncating, but never let an
    # OverflowError ("inf") escape.
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_nonneg_int(value: Any, default: int = 0) -> int:
    """Like ``_safe_int`` but floored at 0.

    Used for every micros field EXCEPT ``subscription_micros`` — the contract
    says subscription balance is the only value allowed to be negative (debt).
    A server-sent negative on any other field is clamped so it can't leak into
    display math.
    """
    return max(0, _safe_int(value, default))


def _safe_bool_str(value: Any) -> bool:
    """Parse an HTTP-header boolean string.

    HTTP headers are ALWAYS strings, so ``paid_access`` arrives as the literal
    string "true" or "false".  ``bool("false")`` is ``True`` in Python — that
    would be a silent never-depleted bug — so we compare the normalised string
    explicitly.  Callers handle the absent (fail-open) case before calling this.
    """
    return str(value).strip().lower() == "true"


@dataclass
class CreditsState:
    """Full credits state parsed from x-nous-credits-* response headers."""

    version: int = 0
    remaining_micros: int = 0
    remaining_usd: str = ""
    subscription_micros: int = 0  # SIGNED — may be negative (debt). ONLY field allowed negative.
    subscription_usd: str = ""
    subscription_limit_micros: Optional[int] = None  # PAIRED + OPTIONAL (only when subscription_cap)
    subscription_limit_usd: Optional[str] = None
    rollover_micros: int = 0
    purchased_micros: int = 0
    purchased_usd: str = ""
    denominator_kind: str = "none"  # "subscription_cap" | "none"
    paid_access: bool = True  # depletion keys off THIS == False, NEVER remaining==0
    disabled_reason: Optional[str] = None  # header omitted entirely when null
    as_of_ms: int = 0
    captured_at: float = 0.0  # time.time() when this was captured

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        if not self.has_data:
            return float("inf")
        return time.time() - self.captured_at

    @property
    def used_fraction(self) -> Optional[float]:
        """Fraction of the subscription cap consumed, in [0.0, 1.0].

        Only computable when the denominator is a subscription cap AND the cap
        is a positive int.  Returns None otherwise (no meaningful denominator).
        """
        if self.denominator_kind != "subscription_cap":
            return None
        if not isinstance(self.subscription_limit_micros, int):
            return None
        if self.subscription_limit_micros <= 0:
            return None
        used = self.subscription_limit_micros - self.subscription_micros
        return max(0.0, min(1.0, used / self.subscription_limit_micros))


def parse_credits_headers(
    headers: Mapping[str, str],
    provider: str = "",
) -> Optional[CreditsState]:
    """Parse x-nous-credits-* headers into a CreditsState.

    Returns None if no credits headers are present.  Fail-open: missing or
    malformed fields fall back to safe defaults rather than raising.
    """
    # Normalize to lowercase so lookups work regardless of how the server
    # capitalises headers (HTTP header names are case-insensitive per RFC 7230).
    lowered = {k.lower(): v for k, v in headers.items()}

    # Quick check: at least one credits header must exist.
    has_any = any(k.startswith("x-nous-credits-") for k in lowered)
    if not has_any:
        return None

    # paid_access: absent header -> True (fail-open: assume access unless told
    # otherwise); present -> parse the "true"/"false" string explicitly.
    if "x-nous-credits-paid-access" in lowered:
        paid_access = _safe_bool_str(lowered.get("x-nous-credits-paid-access"))
    else:
        paid_access = True

    # disabled_reason: header omitted entirely when null — keep None, do NOT
    # coerce to empty string.
    disabled_reason = lowered.get("x-nous-credits-disabled-reason")

    # subscription_limit_*: PAIRED. If exactly one side is present, the pair is
    # half-formed — reject both rather than emit a misleading partial cap.
    sub_limit_micros_raw = lowered.get("x-nous-credits-subscription-limit-micros")
    sub_limit_usd_raw = lowered.get("x-nous-credits-subscription-limit-usd")
    if sub_limit_micros_raw is not None and sub_limit_usd_raw is not None:
        subscription_limit_micros: Optional[int] = _safe_nonneg_int(sub_limit_micros_raw)
        subscription_limit_usd: Optional[str] = sub_limit_usd_raw
    else:
        subscription_limit_micros = None
        subscription_limit_usd = None

    return CreditsState(
        version=_safe_nonneg_int(lowered.get("x-nous-credits-version")),
        remaining_micros=_safe_nonneg_int(lowered.get("x-nous-credits-remaining-micros")),
        remaining_usd=lowered.get("x-nous-credits-remaining-usd", ""),
        subscription_micros=_safe_int(lowered.get("x-nous-credits-subscription-micros")),
        subscription_usd=lowered.get("x-nous-credits-subscription-usd", ""),
        subscription_limit_micros=subscription_limit_micros,
        subscription_limit_usd=subscription_limit_usd,
        rollover_micros=_safe_nonneg_int(lowered.get("x-nous-credits-rollover-micros")),
        purchased_micros=_safe_nonneg_int(lowered.get("x-nous-credits-purchased-micros")),
        purchased_usd=lowered.get("x-nous-credits-purchased-usd", ""),
        denominator_kind=lowered.get("x-nous-credits-denominator-kind", "none"),
        paid_access=paid_access,
        disabled_reason=disabled_reason,
        as_of_ms=_safe_nonneg_int(lowered.get("x-nous-credits-as-of-ms")),
        captured_at=time.time(),
    )
