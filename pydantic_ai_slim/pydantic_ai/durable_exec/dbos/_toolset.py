from __future__ import annotations

from typing import Any, cast

from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import Tool


class DBOSWorkflowConfig(TypedDict, total=False):
    """Options for running a tool as its own DBOS workflow, instead of inline in the caller's.

    Set via tool metadata: `metadata={'dbos_workflow': True}` (defaults) or
    `metadata={'dbos_workflow': DBOSWorkflowConfig(...)}`. `max_recovery_attempts` maps to the
    `@DBOS.workflow(...)` decorator option of the same name.
    """

    max_recovery_attempts: int


def resolve_tool_workflow_config(
    metadata: dict[str, Any] | None,
    tool_name: str,
) -> DBOSWorkflowConfig | None:
    """Resolve the per-tool DBOS-workflow config, or `None` when the tool doesn't get one.

    Two metadata keys feed this, in order of precedence:

    1. `'dbos_workflow'` (`True` or a [`DBOSWorkflowConfig`][pydantic_ai.durable_exec.dbos.DBOSWorkflowConfig]
       dict) -- explicit, DBOS-specific opt-in, normally set by whoever configures a *specific*
       agent's durability.
    2. `'nested_agent_run'` (`True`) -- an engine-neutral fact a toolset author declares about
       their *own* tool: "this tool's body runs another agent (`agent.run()`)". Set by the tool's
       author (e.g. a delegation/sub-agent toolset), who has no reason to know or care which
       durability engine, if any, wraps the outer agent. `DBOSDurability` treats this as "give the
       nested run its own DBOS workflow" -- the natural DBOS-specific consequence of that fact,
       decided *here*, not by the toolset: without it, DBOS would otherwise flatten the nested
       `agent.run()` into the caller's own workflow (see `_install_workflow_wrappers`), losing the
       sub-agent's independent identity, observability, and recoverability as a DBOS workflow.

    An explicit `'dbos_workflow'` always wins over a `'nested_agent_run'` hint (e.g. to give it a
    non-default `max_recovery_attempts`).
    """
    if metadata is None:
        return None
    workflow_config = metadata.get('dbos_workflow')
    if workflow_config is None:
        if metadata.get('nested_agent_run'):
            return DBOSWorkflowConfig()
        return None
    if workflow_config is True:
        return DBOSWorkflowConfig()
    if isinstance(workflow_config, dict):
        return cast('DBOSWorkflowConfig', workflow_config)
    raise UserError(
        f"Tool {tool_name!r} has invalid 'dbos_workflow' metadata: expected `True` or a dict "
        f'(`DBOSWorkflowConfig`), got {type(workflow_config).__name__}.'
    )


def toolset_needs_dbos_wrapping(tools: dict[str, Tool[Any]]) -> bool:
    """Whether any tool in a `FunctionToolset.tools` mapping needs DBOS-workflow wrapping.

    Checked at bind time (`DBOSDurability.for_agent`) against the plain `Tool` objects a
    `FunctionToolset` already holds, so toolsets with no tagged tool are left completely
    unwrapped -- same behaviour, same lack of an `id` requirement, as before this feature existed.
    """
    return any(resolve_tool_workflow_config(tool.metadata, name) is not None for name, tool in tools.items())
