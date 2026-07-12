"""Internal building blocks shared by the bundled `durable_exec` integrations.

Not public API. The surface third-party durable-execution integrations should
build on is the wrapper hierarchy ([`WrapperAgent`][pydantic_ai.agent.WrapperAgent]
/ [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel] /
[`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset]) plus the
[`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] hooks.
A first-class integration surface for runtimes is tracked as
[#5477](https://github.com/pydantic/pydantic-ai/issues/5477); until then these
helpers are reserved for the bundled `temporal`, `dbos`, and `prefect`
integrations.
"""

from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

from pydantic_ai._utils import disable_threads
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

__all__ = [
    'DurableModel',
    'SegmentExecutor',
    'StreamedActivityResult',
    'disable_threads',
    'capture_event_stream',
    'unwrap_model',
    'DurableRunContext',
    'strip_run_context',
]


def unwrap_model(model: Model) -> Model:
    """Strip [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel] layers to the underlying model.

    Durability capabilities close over the agent's construction-time model and need to
    detect when a *different* model is supplied at run time (via `run(model=...)` /
    `override(model=...)`). Comparing `model_id` strings is too coarse — two distinct
    instances (e.g. the same model name on different providers, base URLs, or API keys)
    share a `model_id` — while comparing the wrapped instances directly is too strict,
    because an [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] capability
    wraps the model in an [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel]
    before the request runs. Unwrapping both sides and comparing by identity gets it
    right: a normal run's instrumented model unwraps to the same underlying instance,
    while a genuine runtime override unwraps to a different one.
    """
    while isinstance(model, WrapperModel):
        model = model.wrapped
    return model


@dataclass
class StreamedActivityResult:
    """Bundle returned across an activity/step/task boundary in durable-execution flows.

    Carries both the final `ModelResponse` and the raw events captured from the live
    model stream inside the boundary. The chain consumes the replayed events workflow-side.
    This is the serializable counterpart of a
    [`CompletedStreamedResponse`][pydantic_ai.models.CompletedStreamedResponse].
    """

    response: ModelResponse
    events: list[ModelResponseStreamEvent]


_ResultT = TypeVar('_ResultT')

SegmentExecutor: TypeAlias = Callable[
    [list[ModelMessage], 'ModelSettings | None', ModelRequestParameters], Awaitable[_ResultT]
]
"""Executes one model-request segment inside an engine's durable unit (activity/step/task)."""


class DurableModel(WrapperModel):
    """Dispatches each model-request segment through its own durable unit.

    The bundled durability capabilities swap this in for `request_context.model` in
    `wrap_model_request` and run the innermost handler in workflow/flow code, so the
    continuation loop (Anthropic `pause_turn`, OpenAI background mode) checkpoints every
    suspended segment durably and a failed segment retries alone, while everything else
    (`profile`, `settings`, `continuation_delay`, ...) is answered by the wrapped
    workflow-side model. Everything engine-specific lives in the three executors, each
    running one request / streamed request / cancellation inside the engine's
    activity, step, or task.
    """

    def __init__(
        self,
        wrapped: Model,
        *,
        request_segment: SegmentExecutor[ModelResponse],
        request_stream_segment: SegmentExecutor[StreamedActivityResult],
        cancel_suspended_response_segment: Callable[[ModelResponse], Awaitable[None]],
    ):
        super().__init__(wrapped)
        self._request_segment = request_segment
        self._request_stream_segment = request_stream_segment
        self._cancel_suspended_response_segment = cancel_suspended_response_segment

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self._request_segment(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[CompletedStreamedResponse]:
        result = await self._request_stream_segment(messages, model_settings, model_request_parameters)
        yield CompletedStreamedResponse(
            result.response,
            model_request_parameters=model_request_parameters,
            events=result.events,
        )

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        await self._cancel_suspended_response_segment(response)


async def capture_event_stream(
    *,
    run_context: RunContext[Any],
    stream: AsyncIterable[ModelResponseStreamEvent],
    handler: EventStreamHandler[Any] | None,
) -> list[ModelResponseStreamEvent]:
    """Capture a live model stream inside a durable-execution boundary.

    If a handler is provided, it consumes the live stream inside the boundary. Any
    events it leaves unconsumed are drained and captured. The returned raw events are
    shipped back to the workflow, where the capability chain and any per-run handler
    consume the replay.

    Args:
        run_context: The current agent run context.
        stream: The live model stream.
        handler: Optional handler to run inside the durable boundary.
    """
    captured: list[ModelResponseStreamEvent] = []

    async def teed() -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in stream:
            captured.append(event)
            yield event

    teed_stream = teed()
    if handler is not None:
        await handler(run_context, teed_stream)

    async for _ in teed_stream:
        pass
    return captured


_DURABLE_RUN_CONTEXT_FIELDS = (
    'run_id',
    'metadata',
    'retries',
    'tool_call_id',
    'tool_name',
    'tool_call_approved',
    'tool_call_metadata',
    'retry',
    'max_retries',
    'run_step',
    'partial_output',
    'usage',
    'usage_limits',
    'loaded_capability_ids',
    'discovered_tool_names',
    'capability_loaded',
)
"""Fields carried across a durable-execution boundary by `strip_run_context`/`DurableRunContext`.

Mirrors the field set `TemporalRunContext.serialize_run_context` exposes across its JSON
activity boundary (`durable_exec/temporal/_run_context.py`) -- deliberately the same list, kept
in sync by hand, so a tool that works across the Temporal boundary works the same way across the
DBOS/Prefect one. Excludes `agent` (holds the live agent -- toolsets, models, capabilities --
none of it durably picklable/JSON-safe) and `model`/`tracer`/`prompt`/`messages`/
`validation_context`/`model_settings`/`conversation_id`/`trace_include_content`/
`instrumentation_version` for the same reason Temporal excludes them: not needed by the tool
bodies this is built for (which read `deps`/`usage`/metadata, not the run's model or trace
context), and either unpicklable/not JSON-safe or simply redundant this deep in a tool call.
"""


class DurableRunContext(RunContext[Any]):
    """A `RunContext` reconstructed from a `strip_run_context` snapshot inside a durable boundary.

    Only the fields in `_DURABLE_RUN_CONTEXT_FIELDS` (plus `deps`, always required) are
    available; `agent` is `None` unless attached explicitly by the caller after construction (the
    DBOS/Prefect wrapper toolsets do this from a live reference kept outside what got persisted,
    not from anything serialized -- the same two-step Temporal's `deserialize_run_context` does).
    Accessing any other `RunContext` field raises a clear `UserError` instead of `AttributeError`.
    """

    def __init__(self, deps: Any, **kwargs: Any) -> None:
        self.__dict__ = {**kwargs, 'deps': deps}
        self.__dict__.setdefault('agent', None)
        self._set_dataclass_fields()

    def _set_dataclass_fields(self) -> None:
        # Restricting `__dataclass_fields__` to the populated subset is what makes
        # `dataclasses.replace()` (used elsewhere in the tool-call chain) work against a partial
        # object instead of erroring on the fields `strip_run_context` dropped. Kept out of
        # `__getstate__`/pickled state (below): a `dataclasses.Field`'s `metadata` is a
        # `mappingproxy`, which the `pickle` module DBOS/Prefect use for durable persistence can't
        # serialize -- recomputed on the unpickled side instead.
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in self.__dict__},
        )

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != '__dataclass_fields__'}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self._set_dataclass_fields()

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            if name in RunContext.__dataclass_fields__:
                raise UserError(
                    f'{self.__class__.__name__!r} object has no attribute {name!r}: it was dropped by '
                    '`strip_run_context` when crossing the durable-execution boundary.'
                ) from e
            else:
                raise e


def strip_run_context(ctx: RunContext[Any]) -> DurableRunContext:
    """Snapshot `ctx` into a `DurableRunContext`, dropping fields unsafe to persist durably.

    Use before handing a `RunContext` to a DBOS workflow or Prefect (sub)flow: unlike a Temporal
    activity (JSON-serialized explicitly) or a DBOS/Prefect *task* (pickled but not durably
    persisted as replayable input), a DBOS workflow or Prefect flow run persists its call
    arguments for recovery/observability, so a live `ctx.agent` (unpicklable: it reaches back
    into toolsets, models, and capabilities) would either fail to pickle or bloat every recorded
    run with the whole agent graph.

    The result's `agent` is always `None`; attach a real one with
    `result.__dict__['agent'] = agent` (and `result.__dict__['root_capability'] = agent.root_capability`
    if the chain needs to run against it) from a reference kept outside the durable input, the way
    Temporal's `deserialize_run_context` does for its own activity boundary.
    """
    return DurableRunContext(deps=ctx.deps, **{name: getattr(ctx, name) for name in _DURABLE_RUN_CONTEXT_FIELDS})
