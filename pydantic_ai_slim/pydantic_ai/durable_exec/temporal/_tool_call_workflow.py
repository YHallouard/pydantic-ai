from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, with_config
from temporalio import workflow

from pydantic_ai.exceptions import UserError

from ._toolset import CallToolParams, CallToolResult

if TYPE_CHECKING:
    from ._durability import TemporalDurability


_TOOL_CALL_TARGETS: dict[str, TemporalDurability[Any]] = {}
"""Bound `TemporalDurability` instances by activity name prefix (`agent__{agent.name}`).

Populated at worker setup by `PydanticAIPlugin`/`AgentPlugin` (the same walk that registers the
durability's activities), read by `ToolCallWorkflow.run` to resolve which agent's toolset serves
an incoming tool call. Module-level so it is shared with sandboxed workflow code: `pydantic_ai`
is a sandbox passthrough module, so the worker-setup writes are visible from inside workflows.
"""


def register_tool_call_target(durability: TemporalDurability[Any]) -> None:
    """Make a bound `TemporalDurability`'s toolsets callable from `ToolCallWorkflow`.

    Called by `PydanticAIPlugin`/`AgentPlugin` during worker configuration for every agent they
    discover; users don't call this directly. `durability` must be the *bound* copy
    (`TemporalDurability.from_agent(agent)`), the one whose temporalized toolsets exist.
    """
    _TOOL_CALL_TARGETS[f'agent__{durability.name}'] = durability


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class ToolCallWorkflowParams:
    """Serializable input of `ToolCallWorkflow`: which agent/toolset to resolve, and the call itself."""

    prefix: str
    toolset_id: str
    call: CallToolParams


@workflow.defn(name='pydantic_ai__tool_call')
class ToolCallWorkflow:
    """Generic workflow executing a single tool call as a child workflow.

    The child-workflow counterpart of the per-toolset `call_tool` activity: same
    `CallToolParams`/`CallToolResult` contract, but the tool's function body runs as *workflow
    code* instead of inside an activity. Any I/O the body performs -- most notably a nested
    `agent.run()` on an agent carrying its own `TemporalDurability` -- becomes activities of this
    child workflow, so a long-running tool doesn't collapse into a single activity and doesn't
    bloat the parent workflow's history.

    Started automatically by `TemporalFunctionToolset` for tools tagged
    `metadata={'temporal_child_workflow': ...}`; registered on the worker automatically by
    `PydanticAIPlugin`/`AgentPlugin`. One module-level class shared by all agents: a class created
    dynamically per agent would not survive the workflow sandbox's re-import of its defining
    module.
    """

    @workflow.run
    async def run(self, params: ToolCallWorkflowParams, deps: Any = None) -> CallToolResult:
        durability = _TOOL_CALL_TARGETS.get(params.prefix)
        if durability is None:
            raise UserError(
                f'No agent registered for tool-call workflow target {params.prefix!r}. '
                'The worker running this child workflow must be configured with the same '
                '`PydanticAIPlugin`/`AgentPlugin` (and agents) as the parent workflow, '
                "so the agent's toolsets are available here."
            )
        return await durability.call_tool_in_workflow(params.toolset_id, params.call, deps)
