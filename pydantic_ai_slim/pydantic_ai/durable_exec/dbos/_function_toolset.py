from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from dbos import DBOS

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.durable_exec._utils import strip_run_context
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._toolset import DBOSWorkflowConfig, resolve_tool_workflow_config

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


class DBOSFunctionToolset(WrapperToolset[AgentDepsT]):
    """Wrap a `FunctionToolset`, giving tools tagged `nested_agent_run` their own DBOS workflow.

    Tagging is done via `metadata={'nested_agent_run': True}` (or explicitly `metadata={'dbos_workflow': ...}`),
    which gives the tool its own DBOS workflow instead of the caller's default behaviour of
    flattening a nested `agent.run()` into the current one (see `_install_workflow_wrappers`).

    Untagged tools are untouched: `call_tool` falls straight through to the wrapped toolset,
    exactly as an unwrapped `FunctionToolset` would run under `DBOSDurability` -- DBOS has always
    run function tools inline, and this wrapper only changes that for tools that opt in.
    """

    def __init__(
        self,
        toolset: FunctionToolset[AgentDepsT],
        *,
        name_prefix: str,
        agent: AbstractAgent[AgentDepsT, Any] | None = None,
    ) -> None:
        super().__init__(toolset)
        self._name_prefix = name_prefix
        self._agent = agent
        self._workflows: dict[str, Callable[[dict[str, Any], RunContext[AgentDepsT]], Awaitable[Any]]] = {}

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Mirrors TemporalWrapperToolset/PrefectWrapperToolset: DBOS-wrapped toolsets cannot be
        # swapped out after the fact.
        return self

    def _get_or_build_workflow(
        self, name: str, config: DBOSWorkflowConfig
    ) -> Callable[[dict[str, Any], RunContext[AgentDepsT]], Awaitable[Any]]:
        cached = self._workflows.get(name)
        if cached is not None:
            return cached

        wrapped = self.wrapped
        agent = self._agent
        workflow_kwargs: dict[str, Any] = {}
        if 'max_recovery_attempts' in config:
            workflow_kwargs['max_recovery_attempts'] = config['max_recovery_attempts']

        @DBOS.workflow(name=f'{self._name_prefix}__{name}__tool_call', **workflow_kwargs)
        async def _run(tool_args: dict[str, Any], stripped_ctx: RunContext[AgentDepsT]) -> Any:
            # `agent` is attached here, from the reference this closure captured at bind time --
            # not from `stripped_ctx`, which is what DBOS durably persisted as this workflow's
            # input. See `strip_run_context` for why `ctx.agent` can't cross this boundary as-is.
            if agent is not None:
                stripped_ctx.__dict__['agent'] = agent
                stripped_ctx.__dict__['root_capability'] = agent.root_capability
            try:
                tool = (await wrapped.get_tools(stripped_ctx))[name]
            except KeyError as e:  # pragma: no cover
                raise UserError(
                    f'Tool {name!r} not found in toolset {self.id!r}. '
                    'Removing or renaming tools during an agent run is not supported with DBOS.'
                ) from e
            # `tool_args` will already have been validated into their proper types by the
            # `ToolManager`, but the round trip through DBOS's pickled workflow input turns them
            # into simple Python types again, so they need re-validating -- same rationale as the
            # Temporal/Prefect equivalents.
            args_dict = tool.args_validator.validate_python(tool_args)
            return await wrapped.call_tool(name, args_dict, stripped_ctx, tool)

        self._workflows[name] = _run
        return _run

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        config = resolve_tool_workflow_config(tool.tool_def.metadata, name)
        if config is None or DBOS.workflow_id is None:
            # Untagged, or called outside any DBOS workflow (e.g. a direct toolset test) -- run
            # inline exactly as an unwrapped `FunctionToolset` would.
            return await super().call_tool(name, tool_args, ctx, tool)

        assert isinstance(tool, FunctionToolsetTool)
        if not tool.is_async:
            raise UserError(
                f'Tool {name!r} is tagged to run as its own DBOS workflow, but DBOS runs non-async '
                'tools inline; make the tool function async instead.'
            )

        workflow_fn = self._get_or_build_workflow(name, config)
        return await workflow_fn(tool_args, strip_run_context(ctx))
