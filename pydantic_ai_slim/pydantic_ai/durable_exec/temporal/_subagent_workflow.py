from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict, with_config
from temporalio import workflow

from pydantic_ai import usage as _usage
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.durable_exec import AgentCarryOver
from pydantic_ai.exceptions import UserError

from ._workflow import PydanticAIWorkflow

__all__ = ['SubAgentRunParams', 'SubAgentRunResult', 'SubAgentWorkflow']

_nested_agents: dict[str, AbstractAgent[Any, Any]] = {}
"""Process-global registry of agents made available to `SubAgentWorkflow`, populated by
`TemporalDurability.for_agent()` when constructed with `nested_agents=[...]`. Registration only
ever happens at worker setup time (agent construction), never inside a workflow -- mirrors the
existing constraint on `TemporalDurability` itself."""


def register_nested_agent(agent: AbstractAgent[Any, Any]) -> None:
    """Register `agent` so `SubAgentWorkflow` can look it up by name.

    Not for direct use -- called by `TemporalDurability.for_agent()` for each of its
    `nested_agents`.
    """
    if not agent.name:
        raise UserError(
            'A nested agent passed to `TemporalDurability(nested_agents=[...])` needs a unique '
            '`name`, same as the parent agent -- it identifies the agent for `SubAgentWorkflow` to '
            'look up.'
        )
    _nested_agents[agent.name] = agent


def resolve_nested_agent(agent_name: str) -> AbstractAgent[Any, Any]:
    """Look up a `nested_agents`-registered agent by name, or raise `UserError`.

    Not for direct use -- called by `SubAgentWorkflow.run()`.
    """
    try:
        return _nested_agents[agent_name]
    except KeyError:
        raise UserError(
            f'No nested agent registered under the name {agent_name!r}. Pass it to the parent '
            "agent's `TemporalDurability(nested_agents=[...])` at construction time, before any "
            'workflow runs.'
        ) from None


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class SubAgentRunParams:
    """Arguments for a `SubAgentWorkflow` run.

    What a parent workflow's delegate tool passes to
    `workflow.execute_child_workflow(SubAgentWorkflow.run, ...)`.
    """

    agent_name: str
    """Name of the nested agent to run, as registered via `TemporalDurability(nested_agents=[...])`."""

    task: str
    """The task/prompt for the nested agent."""

    deps: Any
    """Dependencies for the nested agent's run. Same serialization constraint as any other value
    crossing a Temporal boundary."""

    inherited_metadata: dict[str, Any]
    """Snapshot of the parent run's `RunContext.metadata` to seed the child's own metadata with
    (e.g. a durable environment lease) -- merged into, not replacing, the child agent's configured
    metadata."""


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class SubAgentRunResult:
    """Result of a `SubAgentWorkflow` run."""

    output: str
    usage: _usage.RunUsage


@workflow.defn
class SubAgentWorkflow(PydanticAIWorkflow):
    """Generic child workflow for running a nested (sub-)agent under `TemporalDurability`.

    Started via `workflow.execute_child_workflow(SubAgentWorkflow.run, params, ...)` from a
    delegate tool (which must run inline -- a child workflow can only be started from workflow
    code). Gets its own event-history budget and its own `continue_as_new`
    ([`PydanticAIWorkflow.run_agent`][pydantic_ai.durable_exec.temporal.PydanticAIWorkflow.run_agent]
    handles it the same way it would for a top-level workflow), independent of the parent.

    The nested agent must already be registered via the parent's
    `TemporalDurability(nested_agents=[...])` -- this workflow only looks it up, it never
    constructs one, since `TemporalDurability.for_agent()` (which registers Temporal activities)
    must run at worker setup time, not inside a workflow.
    """

    @workflow.run
    async def run(self, params: SubAgentRunParams, carry_over: AgentCarryOver | None = None) -> SubAgentRunResult:
        agent = resolve_nested_agent(params.agent_name)
        result = await self.run_agent(
            agent,
            params.task,
            deps=params.deps,
            metadata=params.inherited_metadata,
            carry_over=carry_over,
            continue_args=lambda co: (params, co),
        )
        return SubAgentRunResult(output=str(result.output), usage=result.usage)
