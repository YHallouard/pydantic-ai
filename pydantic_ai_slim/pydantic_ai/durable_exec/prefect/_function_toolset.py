from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from prefect import flow, task

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.durable_exec._utils import strip_run_context
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._toolset import PrefectWrapperToolset, resolve_tool_flow_config, resolve_tool_task_config
from ._types import FlowConfig, TaskConfig, default_task_config

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


class PrefectFunctionToolset(PrefectWrapperToolset[AgentDepsT]):
    """A wrapper for FunctionToolset that integrates with Prefect, turning tool calls into Prefect tasks.

    A tool tagged `nested_agent_run` (or explicitly `prefect_subflow`) runs as a Prefect subflow
    instead: its body runs another agent's `run()`, so a plain task -- checkpointed as one opaque
    unit -- would hide that nested run's own model/tool calls from Prefect's UI and retry
    semantics. See `resolve_tool_flow_config`.
    """

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        task_config: TaskConfig,
        tool_task_config: dict[str, TaskConfig | None],
        agent: AbstractAgent[AgentDepsT, Any] | None = None,
    ):
        super().__init__(wrapped)
        self._task_config = default_task_config | (task_config or {})
        self._tool_task_config = tool_task_config or {}
        self._agent = agent
        self._subflows: dict[str, Callable[[dict[str, Any], RunContext[AgentDepsT]], Awaitable[Any]]] = {}

        @task
        async def _call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> Any:
            return await super(PrefectFunctionToolset, self).call_tool(tool_name, tool_args, ctx, tool)

        self._call_tool_task = _call_tool_task

    def _get_or_build_subflow(
        self, name: str, config: FlowConfig
    ) -> Callable[[dict[str, Any], RunContext[AgentDepsT]], Awaitable[Any]]:
        cached = self._subflows.get(name)
        if cached is not None:
            return cached

        wrapped = self.wrapped
        agent = self._agent

        # `validate_parameters=False`: Prefect otherwise builds a Pydantic model from the flow's
        # annotated parameters to validate calls against, which chokes on `RunContext[AgentDepsT]`
        # (an unresolved generic at this point) -- these are internal calls, not user-facing flow
        # parameters, so there's nothing useful to validate.
        @flow(name=f'Delegate: {name}', validate_parameters=False, **config)
        async def _run(tool_args: dict[str, Any], stripped_ctx: RunContext[AgentDepsT]) -> Any:
            # `agent` is attached here, from the reference this closure captured at bind time --
            # not from `stripped_ctx`, which is what Prefect persists as this subflow's input. See
            # `strip_run_context` for why `ctx.agent` can't cross this boundary as-is.
            if agent is not None:
                stripped_ctx.__dict__['agent'] = agent
                stripped_ctx.__dict__['root_capability'] = agent.root_capability
            try:
                tool = (await wrapped.get_tools(stripped_ctx))[name]
            except KeyError as e:  # pragma: no cover
                raise UserError(
                    f'Tool {name!r} not found in toolset {self.id!r}. '
                    'Removing or renaming tools during an agent run is not supported with Prefect.'
                ) from e
            args_dict = tool.args_validator.validate_python(tool_args)
            return await wrapped.call_tool(name, args_dict, stripped_ctx, tool)

        self._subflows[name] = _run
        return _run

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Call a tool, wrapped as a Prefect task or subflow with a descriptive name."""
        flow_config = resolve_tool_flow_config(tool, name)
        if flow_config is not None:
            assert isinstance(tool, FunctionToolsetTool)
            if not tool.is_async:
                raise UserError(
                    f'Tool {name!r} is tagged to run as a Prefect subflow, but non-async tools run '
                    'in threads which are not supported as a flow body; make the tool function async instead.'
                )
            subflow_fn = self._get_or_build_subflow(name, flow_config)
            return await subflow_fn(tool_args, strip_run_context(ctx))

        # Per-tool config comes from `metadata={'prefect': ...}` first, then the deprecated
        # `PrefectAgent`'s by-name dict. `False` disables task wrapping for this tool.
        tool_task_config = resolve_tool_task_config(tool, name, self._tool_task_config)
        if tool_task_config is False:
            return await super().call_tool(name, tool_args, ctx, tool)

        merged_config = self._task_config | tool_task_config

        return await self._call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(
            name, tool_args, ctx, tool
        )
