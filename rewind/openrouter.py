"""Inspect model resolution with OpenRouter app attribution.

Real runs go through Inspect's OpenRouter provider (`openrouter/<provider>/<model>`,
key in `OPENROUTER_API_KEY`, needs the `openrouter` extra: `uv sync --extra
openrouter`). For those models every request carries OpenRouter's attribution
headers (https://openrouter.ai/docs/features/app-attribution):

- `HTTP-Referer`: `$OPENROUTER_APP_URL` (default https://github.com/Neolithic-ai/Rewind)
- `X-OpenRouter-Title` and `X-Title`: `$OPENROUTER_APP_TITLE` (default `Rewind`)

They are passed as the `default_headers` model arg, which Inspect forwards to the
`AsyncOpenAI` client it builds, so they are sent with every request (including
retries). Other models are resolved unchanged.

Cost: Inspect has no prices for most models, but OpenRouter reports the USD cost
of each call in its response (`usage.cost`); `call_cost` reads it back from the
logged call so budgets work without a price table.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from inspect_ai.event import Event, ModelEvent
from inspect_ai.model import Model, get_model

PREFIX = "openrouter/"
KEY_VAR = "OPENROUTER_API_KEY"
DEFAULT_APP_URL = "https://github.com/Neolithic-ai/Rewind"
DEFAULT_APP_TITLE = "Rewind"


def is_openrouter(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIX)


def attribution_headers() -> dict[str, str]:
    url = os.environ.get("OPENROUTER_APP_URL") or DEFAULT_APP_URL
    title = os.environ.get("OPENROUTER_APP_TITLE") or DEFAULT_APP_TITLE
    return {"HTTP-Referer": url, "X-OpenRouter-Title": title, "X-Title": title}


def model_args(model: str) -> dict[str, Any]:
    """Extra `get_model()` args for `model` (attribution headers for OpenRouter)."""
    return {"default_headers": attribution_headers()} if is_openrouter(model) else {}


def require_key(model: str) -> None:
    """Exit with a clear message if an OpenRouter model has no API key."""
    if is_openrouter(model) and not os.environ.get(KEY_VAR):
        raise SystemExit(f"{model}: {KEY_VAR} is not set (OpenRouter models need an API "
                         "key; use --model mockllm/model for a keyless mock run)")


def resolve_model(model: str, **kwargs: Any) -> Model:
    """`get_model(model)`, with attribution headers for `openrouter/...` models."""
    require_key(model)
    return get_model(model, **(model_args(model) | kwargs))


def call_cost(event: ModelEvent) -> float | None:
    """USD cost of one model call: Inspect's own price if it has one, else the
    `usage.cost` OpenRouter reports in every response (kept in the logged call)."""
    if event.output.usage is not None and event.output.usage.total_cost is not None:
        return event.output.usage.total_cost
    response = event.call.response if event.call is not None else None
    usage = response.get("usage") if isinstance(response, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    return float(cost) if isinstance(cost, (int, float)) else None


def events_cost(events: Iterable[Event]) -> float | None:
    """Total USD cost of the model calls in `events` (None if none reported one)."""
    costs = [c for e in events if isinstance(e, ModelEvent) and (c := call_cost(e)) is not None]
    return sum(costs) if costs else None
