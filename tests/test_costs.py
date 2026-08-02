"""Cost accounting must not invent numbers."""
from __future__ import annotations

from app.costs import PRICES_PER_MTOK, RequestCost, Usage, estimate_usage, usage_from_response


class FakeResponse:
    def __init__(self, content: str, usage: dict[str, int] | None = None) -> None:
        self.content = content
        if usage is not None:
            self.usage_metadata = usage


def test_provider_reported_usage_is_not_flagged_as_estimated() -> None:
    usage = usage_from_response(
        FakeResponse("hello", {"input_tokens": 1200, "output_tokens": 300}),
        "mistralai/mistral-nemotron",
    )
    assert usage.input_tokens == 1200 and usage.output_tokens == 300
    assert usage.estimated is False
    # 1200 * 0.15/1e6 + 300 * 0.60/1e6
    assert abs(usage.cost_usd - (0.00018 + 0.00018)) < 1e-9


def test_missing_usage_metadata_falls_back_and_says_so() -> None:
    """A cost that silently mixes measured and guessed tokens is not a cost."""
    usage = usage_from_response(FakeResponse("x" * 400), "mistralai/mistral-nemotron")
    assert usage.estimated is True
    assert usage.output_tokens == 100


def test_unknown_model_reports_zero_cost_but_keeps_tokens() -> None:
    """An obviously-wrong $0.00 beats a plausible number from a neighbour's rate."""
    usage = Usage(model="some/unlisted-model", input_tokens=1000, output_tokens=1000)
    assert usage.cost_usd == 0.0
    assert usage.priced is False
    assert usage.input_tokens == 1000


def test_request_cost_aggregates_every_call() -> None:
    """One answer is several model calls: routing, multi-query, generation."""
    cost = RequestCost()
    cost.record(Usage("meta/llama-3.1-8b-instruct", 500, 50))
    cost.record(Usage("nvidia/nemotron-3-super-120b-a12b", 3000, 200))
    assert cost.input_tokens == 3500
    assert cost.output_tokens == 250
    expected = (500 * 0.05 + 50 * 0.10) / 1e6 + (3000 * 0.60 + 200 * 1.80) / 1e6
    assert abs(cost.total_usd - expected) < 1e-9
    assert cost.estimated is False


def test_one_estimated_call_taints_the_whole_request() -> None:
    cost = RequestCost()
    cost.record(Usage("meta/llama-3.1-8b-instruct", 500, 50, estimated=False))
    cost.record(estimate_usage("mistralai/mistral-nemotron", "p" * 400, "c" * 400))
    assert cost.estimated is True


def test_unpriced_models_are_surfaced_not_hidden() -> None:
    cost = RequestCost()
    cost.record(Usage("some/unlisted-model", 100, 100))
    assert cost.unpriced_models == ["some/unlisted-model"]
    assert cost.summary()["cost_usd"] == 0.0


def test_configured_models_are_all_priced() -> None:
    """A model in use with no price silently reports every request as free."""
    from app.retrieval import graphrag

    for model in (graphrag.MAP_MODEL, graphrag.REDUCE_MODEL):
        assert model in PRICES_PER_MTOK, f"{model} is used but unpriced"
