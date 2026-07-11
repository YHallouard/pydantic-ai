"""Carry-over state for pausing and resuming an agent run across a durable-execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict, with_config

from pydantic_ai import messages as _messages, usage as _usage

__all__ = ['AgentCarryOver', 'AgentRunPaused']


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class AgentCarryOver:
    """State needed to resume a paused agent run in a fresh workflow/step/task run.

    Built by a durability capability (e.g.
    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]) when it decides
    a run should pause rather than continue in the current workflow/step/task run — for Temporal,
    ahead of `workflow.continue_as_new`. Carried by
    [`AgentRunPaused`][pydantic_ai.durable_exec.AgentRunPaused] to the caller, which is
    responsible for resuming the run with this state (e.g. via `message_history=carry_over.messages`)
    in the new run.
    """

    messages: list[_messages.ModelMessage]
    """Message history accumulated so far, to resume with as `message_history`."""

    usage: _usage.RunUsage
    """Usage accumulated so far, to resume with as `usage`."""

    capability_state: dict[str, Any]
    """Per-capability state contributed by `AbstractCapability.on_continue_as_new`, keyed by capability id."""

    metadata: dict[str, Any]
    """Snapshot of `RunContext.metadata` at the pause point (e.g. a durable environment lease)."""


class AgentRunPaused(Exception):
    """Exception to raise (typically from a `wrap_model_request` hook) to pause an agent run.

    Not a failure: a durability capability raises this to signal that the run should stop and be
    resumed in a fresh run (e.g. via Temporal's `continue_as_new`) rather than continue in the
    current one. The caller — typically
    [`PydanticAIWorkflow.run_agent`][pydantic_ai.durable_exec.temporal.PydanticAIWorkflow.run_agent],
    or user workflow code driving `agent.run()` directly — is responsible for catching it and
    starting a new run seeded with `carry_over`.

    Must never be routed through capability error-recovery hooks (`on_model_request_error`,
    `on_run_error`, etc.) — like `asyncio.CancelledError`, always re-raise it if caught generically.
    """

    carry_over: AgentCarryOver

    def __init__(self, carry_over: AgentCarryOver):
        self.carry_over = carry_over
        super().__init__()
