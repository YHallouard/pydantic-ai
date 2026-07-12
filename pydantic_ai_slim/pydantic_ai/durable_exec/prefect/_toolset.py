from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT

from ._types import FlowConfig, TaskConfig

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


def resolve_tool_task_config(
    tool: ToolsetTool[Any] | None,
    tool_name: str,
    tool_task_config: Mapping[str, TaskConfig | None],
) -> TaskConfig | Literal[False]:
    """Resolve per-tool Prefect task config.

    Reads `tool.tool_def.metadata['prefect']` first, then falls back to the explicit
    `tool_task_config` dict keyed by tool name. Returns a `TaskConfig` dict (possibly
    empty), or `False` to skip task wrapping.
    """
    # Metadata set on the tool (via @toolset.tool(metadata={'prefect': ...}), with_metadata, or
    # the `SetToolMetadata` capability) is the primary path.
    if tool is not None and tool.tool_def.metadata is not None:
        metadata_config = tool.tool_def.metadata.get('prefect')
        if metadata_config is False:
            return False
        if metadata_config is not None:
            if not isinstance(metadata_config, dict):
                raise UserError(
                    f"Tool {tool_name!r} has invalid 'prefect' metadata: expected a dict "
                    f'(`TaskConfig`) or `False`, got {type(metadata_config).__name__}.'
                )
            return cast('TaskConfig', metadata_config)
    # Fallback: per-tool dict passed to the deprecated `PrefectAgent`. An explicit `None`
    # disables wrapping; a missing key means "use the base config".
    if tool_name in tool_task_config:
        fallback = tool_task_config[tool_name]
        return False if fallback is None else fallback
    return {}


def resolve_tool_flow_config(
    tool: ToolsetTool[Any] | None,
    tool_name: str,
) -> FlowConfig | None:
    """Resolve the per-tool subflow config, or `None` when the tool doesn't get one.

    Two metadata keys feed this, in order of precedence:

    1. `'prefect_subflow'` (`True` or a [`FlowConfig`][pydantic_ai.durable_exec.prefect.FlowConfig]
       dict) -- explicit, Prefect-specific opt-in, normally set by whoever configures a *specific*
       agent's durability (retries, timeouts, etc. are deployment concerns), not by the toolset author.
    2. `'nested_agent_run'` (`True`) -- an engine-neutral fact a toolset author declares about
       their *own* tool: "this tool's body runs another agent (`agent.run()`)". Set by the tool's
       author (e.g. a delegation/sub-agent toolset), who has no reason to know or care which
       durability engine, if any, wraps the outer agent. `PrefectDurability` treats this as "run
       as a subflow instead of a task" -- the natural Prefect-specific consequence of that fact,
       decided *here*, not by the toolset. A future DBOS durability capability interprets the same
       tag differently (its own workflow, not a subflow-vs-task distinction -- DBOS doesn't wrap
       tools in anything by default).

    An explicit `'prefect_subflow'` always wins over a `'nested_agent_run'` hint (e.g. to give it
    a longer `timeout_seconds`).
    """
    if tool is None or tool.tool_def.metadata is None:
        return None
    metadata = tool.tool_def.metadata
    flow_config = metadata.get('prefect_subflow')
    if flow_config is None:
        if metadata.get('nested_agent_run'):
            return FlowConfig()
        return None
    if flow_config is True:
        return FlowConfig()
    if isinstance(flow_config, dict):
        return cast('FlowConfig', flow_config)
    raise UserError(
        f"Tool {tool_name!r} has invalid 'prefect_subflow' metadata: expected `True` or a dict "
        f'(`FlowConfig`), got {type(flow_config).__name__}.'
    )


class PrefectWrapperToolset(WrapperToolset[AgentDepsT], ABC):
    """Base class for Prefect-wrapped toolsets."""

    @property
    def id(self) -> str | None:
        # Prefect toolsets should have IDs for better task naming
        return self.wrapped.id

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Prefect-ified toolsets cannot be swapped out after the fact.
        return self


def prefectify_toolset(
    toolset: AbstractToolset[AgentDepsT],
    mcp_task_config: TaskConfig,
    tool_task_config: TaskConfig,
    tool_task_config_by_name: dict[str, TaskConfig | None],
    agent: AbstractAgent[AgentDepsT, Any] | None = None,
) -> AbstractToolset[AgentDepsT]:
    """Wrap a toolset to integrate it with Prefect.

    Args:
        toolset: The toolset to wrap.
        mcp_task_config: The Prefect task config to use for MCP server tasks.
        tool_task_config: The default Prefect task config to use for tool calls.
        tool_task_config_by_name: Per-tool task configuration. Keys are tool names, values are TaskConfig or None.
        agent: The agent instance to attach to the run context reconstructed inside a
            `nested_agent_run`/`prefect_subflow`-tagged tool's subflow.
    """
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import PrefectFunctionToolset

        return PrefectFunctionToolset(
            wrapped=toolset,
            task_config=tool_task_config,
            tool_task_config=tool_task_config_by_name,
            agent=agent,
        )

    try:
        from pydantic_ai.mcp import MCPToolset

        from ._mcp_toolset import PrefectMCPToolset
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPToolset):
            return PrefectMCPToolset(
                wrapped=toolset,
                task_config=mcp_task_config,
            )

    return toolset
