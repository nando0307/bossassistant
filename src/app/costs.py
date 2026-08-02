"""Per-request token and cost accounting.

"$0.004 per query at p50 1.2s" is the pair of numbers an enterprise buyer asks
for first, and almost no portfolio project can produce either. Latency this
project already measures; this adds the other half.

Two honesty constraints shape the implementation:

* **Prices are declared, not discovered.** NVIDIA's hosted endpoints do not
  return a price, so the table below is a local assumption that goes stale. It
  is dated, and unknown models are reported at zero cost with the token counts
  intact — an obviously-wrong $0.00 is better than a plausible number invented
  from a neighbouring model's rate.
* **Token counts come from the provider when it reports them.** `usage_metadata`
  is authoritative; the character-based fallback is an estimate and is labelled
  as one, because a cost figure that silently mixes measured and guessed tokens
  is not a cost figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: USD per 1M tokens, as published for NVIDIA-hosted endpoints.
#: Checked 2026-08-02. Re-check before quoting these anywhere that matters.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "nvidia/nemotron-3-super-120b-a12b": (0.60, 1.80),
    "nvidia/nemotron-3-ultra-550b-a55b": (1.20, 3.60),
    "mistralai/mistral-nemotron": (0.15, 0.60),
    "meta/llama-3.1-8b-instruct": (0.05, 0.10),
    "meta/llama-3.3-70b-instruct": (0.35, 0.70),
    "nvidia/nv-embedqa-e5-v5": (0.02, 0.0),
}

#: Rough bytes-per-token for English prose, used only when the provider omits
#: usage metadata. Deliberately crude: precision here would imply the number is
#: measured, and it is not.
CHARS_PER_TOKEN = 4.0


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    #: True when any component was estimated rather than reported.
    estimated: bool = False

    @property
    def cost_usd(self) -> float:
        rates = PRICES_PER_MTOK.get(self.model)
        if rates is None:
            return 0.0
        input_rate, output_rate = rates
        return (self.input_tokens * input_rate + self.output_tokens * output_rate) / 1_000_000

    @property
    def priced(self) -> bool:
        return self.model in PRICES_PER_MTOK


@dataclass
class RequestCost:
    """Accumulates every model call made while answering one request."""

    calls: list[Usage] = field(default_factory=list)

    def record(self, usage: Usage) -> None:
        self.calls.append(usage)

    @property
    def total_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    @property
    def estimated(self) -> bool:
        return any(call.estimated for call in self.calls)

    @property
    def unpriced_models(self) -> list[str]:
        return sorted({c.model for c in self.calls if not c.priced})

    def summary(self) -> dict[str, object]:
        return {
            "calls": len(self.calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.total_usd, 6),
            "estimated": self.estimated,
            "unpriced_models": self.unpriced_models,
        }


def usage_from_response(response: object, model: str) -> Usage:
    """Read token counts off a LangChain response, falling back to an estimate.

    LangChain surfaces provider counts on `usage_metadata`. When the provider
    omits them the counts are derived from length and flagged `estimated`, so a
    downstream reader can tell a measured cost from an inferred one.
    """
    metadata = getattr(response, "usage_metadata", None)
    if isinstance(metadata, dict) and metadata.get("input_tokens") is not None:
        return Usage(
            model=model,
            input_tokens=int(metadata.get("input_tokens") or 0),
            output_tokens=int(metadata.get("output_tokens") or 0),
            estimated=False,
        )
    text = getattr(response, "content", None)
    if not isinstance(text, str):
        text = str(response) if response is not None else ""
    return Usage(
        model=model,
        input_tokens=0,
        output_tokens=int(len(text) / CHARS_PER_TOKEN),
        estimated=True,
    )


def estimate_usage(model: str, prompt: str, completion: str) -> Usage:
    """Estimate both sides from text. Always flagged as estimated."""
    return Usage(
        model=model,
        input_tokens=int(len(prompt) / CHARS_PER_TOKEN),
        output_tokens=int(len(completion) / CHARS_PER_TOKEN),
        estimated=True,
    )
