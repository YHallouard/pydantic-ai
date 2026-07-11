from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from temporalio import workflow

from pydantic_ai import _instructions, messages as _messages, models, usage as _usage
from pydantic_ai.agent.abstract import (
    AbstractAgent,
    AgentMetadata,
    AgentModelSettings,
    AgentRetries,
)
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.durable_exec import AgentCarryOver
from pydantic_ai.exceptions import AgentRunPaused, UserError
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, DeferredToolResults
from pydantic_ai.toolsets import AbstractToolset

if TYPE_CHECKING:
    from pydantic_ai.agent.spec import AgentSpec


class PydanticAIWorkflow:
    """Temporal Workflow base class that provides `__pydantic_ai_agents__` for direct agent registration.

    Accepts any `AbstractAgent` — either a regular `Agent` carrying a
    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]
    capability, or the deprecated
    [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent] wrapper.
    [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin]
    walks the sequence and registers each agent's activities with the worker.
    """

    __pydantic_ai_agents__: Sequence[AbstractAgent[Any, Any]]

    async def run_agent(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
        carry_over: AgentCarryOver | None = None,
        continue_args: Callable[[AgentCarryOver], Sequence[Any]] | None = None,
    ) -> AgentRunResult[Any]:
        """Run `agent`, transparently continuing-as-new when the run pauses.

        A thin `agent.run()` wrapper for `@workflow.run` methods that also want
        [`TemporalDurability(continue_as_new='auto')`][pydantic_ai.durable_exec.temporal.TemporalDurability]
        support: catches [`AgentRunPaused`][pydantic_ai.exceptions.AgentRunPaused] and calls
        `workflow.continue_as_new` to restart the workflow with resumed state, instead of letting
        the exception propagate as a workflow failure.

        Args:
            agent: The agent to run.
            user_prompt: User input to start/continue the conversation. Ignored (in favor of
                `carry_over.messages`) when resuming, i.e. when this call's own `carry_over`
                argument is not `None`.
            carry_over: The `AgentCarryOver` this `@workflow.run` method was passed by its own
                `continue_as_new` restart (`None` on a fresh run). When set, seeds
                `message_history`, `usage`, and `metadata` for the resumed run — pass the
                workflow's own `carry_over` parameter straight through; don't destructure it
                yourself.
            continue_args: Required to actually continue-as-new on `AgentRunPaused`. Receives the
                `AgentCarryOver` from the exception and must return the positional arguments to
                restart this same `@workflow.run` method with — typically
                `lambda co: (original_args, co)`, i.e. the same arguments the workflow was
                originally started with, plus the carry-over as a trailing parameter. Without this,
                a pause instead re-raises as `UserError` (matching `TemporalDurability`'s "no code
                to actually restart" case: better a clear error than a silently non-durable run).
            output_type: Forwarded to `agent.run()`. Unlike `agent.run()` itself, a per-call
                override here doesn't narrow the return type (it's always `AgentRunResult[Any]`)
                — this is a thin control-flow wrapper, not a typed proxy.
            message_history: Forwarded to `agent.run()`; overridden by `carry_over.messages` when
                resuming.
            deferred_tool_results: Forwarded to `agent.run()`.
            conversation_id: Forwarded to `agent.run()`.
            model: Forwarded to `agent.run()`.
            instructions: Forwarded to `agent.run()`.
            deps: Forwarded to `agent.run()`.
            model_settings: Forwarded to `agent.run()`.
            usage_limits: Forwarded to `agent.run()`.
            usage: Forwarded to `agent.run()`; overridden by `carry_over.usage` when resuming.
            metadata: Forwarded to `agent.run()`; overridden by `carry_over.metadata` when resuming.
            retries: Forwarded to `agent.run()`.
            toolsets: Forwarded to `agent.run()`.
            capabilities: Forwarded to `agent.run()`.
            spec: Forwarded to `agent.run()`.

        Returns:
            The result of the run, once it completes without pausing.
        """
        if carry_over is not None:
            message_history = carry_over.messages
            usage = carry_over.usage
            metadata = carry_over.metadata

        try:
            return await agent.run(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                retries=retries,
                toolsets=toolsets,
                capabilities=capabilities,
                spec=spec,
            )
        except AgentRunPaused as e:
            if continue_args is None:
                raise UserError(
                    'Agent run paused for continue-as-new, but `run_agent()` was not given a '
                    "`continue_args` callable to rebuild this `@workflow.run` method's arguments "
                    'from the `AgentCarryOver`. Pass one (or set `continue_as_new=False` on the '
                    'TemporalDurability capability to disable pausing).'
                ) from e
            workflow.continue_as_new(args=list(continue_args(e.carry_over)))
