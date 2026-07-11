"""Carry-over state for pausing and resuming an agent run across a durable-execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict, with_config

from pydantic_ai import messages as _messages, usage as _usage

__all__ = ['AgentCarryOver']


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class AgentCarryOver:
    """State needed to resume a paused agent run in a fresh workflow/step/task run.

    Built by a durability capability (e.g.
    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]) when it decides
    a run should pause rather than continue in the current workflow/step/task run — for Temporal,
    ahead of `workflow.continue_as_new`. Carried by
    [`AgentRunPaused`][pydantic_ai.exceptions.AgentRunPaused] to the caller, which is
    responsible for resuming the run with this state (e.g. via `message_history=carry_over.messages`)
    in the new run.
    """

    messages: list[_messages.ModelMessage]
    """Message history accumulated so far, to resume with as `message_history`."""

    usage: _usage.RunUsage
    """Usage accumulated so far, to resume with as `usage`."""

    metadata: dict[str, Any]
    """Snapshot of `RunContext.metadata` at the pause point.

    The general-purpose bag for any capability's own state that needs to survive the pause (e.g. a
    durable environment lease) — a capability that wants to carry something across the boundary
    reads and writes it here like any other run-scoped state, the same way it already would within
    a single run. No dedicated per-capability hook is needed for this.
    """
