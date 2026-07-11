"""Engine-neutral `durable_exec` tests.

Covers `AgentRunPaused`/`AgentCarryOver`, the shared primitives any durability capability (not
just `TemporalDurability`) can use to signal that a run should pause and resume elsewhere.
`TemporalDurability`'s own continue-as-new detection and `workflow.continue_as_new` restart are
covered in `test_temporal.py`, which needs a running Temporal server.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from collections.abc import AsyncIterable

from pydantic_ai import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.durable_exec import AgentCarryOver
from pydantic_ai.exceptions import AgentRunPaused
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

pytestmark = pytest.mark.anyio


def _unreachable_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise AssertionError('the model should never be called: the pausing capability intercepts the request first')


@dataclass
class _PausingCapability(AbstractCapability):
    """Stands in for a durability capability (like `TemporalDurability`) deciding to pause."""

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        raise AgentRunPaused(AgentCarryOver(messages=list(request_context.messages), usage=ctx.usage, metadata={}))


@dataclass
class _GenericRecoveryCapability(AbstractCapability):
    """Stands in for a capability with broad catch-and-recover error handling (e.g. a
    model-fallback capability) elsewhere in the chain -- the kind of capability that would
    silently turn `AgentRunPaused` into a fake successful response if it were ever allowed to see
    it as a plain `error`."""

    async def on_model_request_error(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart('recovered')])


async def test_agent_run_paused_not_swallowed_by_generic_recovery_capability():
    """`AgentRunPaused` must propagate out of `agent.run()`, not be routed through
    `on_model_request_error` -- even when another capability in the chain overrides that hook for
    generic recovery. Exercises the non-streaming carve-out site (`_make_request`).

    Not reachable via a VCR test: no real model raises `AgentRunPaused`, and no real provider
    response exercises this internal carve-out against an adversarial capability chain. Without
    it, `_GenericRecoveryCapability` would catch the pause and this run would return `'recovered'`
    instead of raising.
    """
    agent = Agent(
        FunctionModel(_unreachable_model),
        capabilities=[_GenericRecoveryCapability(), _PausingCapability()],
    )
    with pytest.raises(AgentRunPaused) as exc_info:
        await agent.run('hello')
    assert exc_info.value.carry_over.messages


async def test_agent_run_paused_not_swallowed_in_streaming_short_circuit():
    """Same as above, but forcing the streaming path's short-circuit carve-out
    (`_resolve_wrap_result`) by setting `event_stream_handler` -- a capability whose
    `wrap_model_request` raises before ever calling the streaming handler short-circuits `stream()`
    without opening a model stream, a different internal branch than the non-streaming case."""

    async def _handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:  # pragma: no cover
        raise AssertionError('the stream handler should never run: wrap_model_request pauses first')

    agent = Agent(
        FunctionModel(_unreachable_model),
        capabilities=[_GenericRecoveryCapability(), _PausingCapability()],
    )
    with pytest.raises(AgentRunPaused) as exc_info:
        await agent.run('hello', event_stream_handler=_handler)
    assert exc_info.value.carry_over.messages
