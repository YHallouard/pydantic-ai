"""Durable execution integrations for Pydantic AI.

Each subpackage adds durability for one durable-execution platform via a
capability you attach to an [`Agent`][pydantic_ai.Agent]:

- [`pydantic_ai.durable_exec.temporal`][pydantic_ai.durable_exec.temporal] —
  [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]
- [`pydantic_ai.durable_exec.dbos`][pydantic_ai.durable_exec.dbos] —
  [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability]
- [`pydantic_ai.durable_exec.prefect`][pydantic_ai.durable_exec.prefect] —
  [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability]

[`AgentCarryOver`][pydantic_ai.durable_exec.AgentCarryOver] and
[`AgentRunPaused`][pydantic_ai.exceptions.AgentRunPaused] are shared across engines: any
durability capability can raise `AgentRunPaused` to signal that a run should stop and resume in a
fresh run rather than continue in the current one.
"""

from ..exceptions import AgentRunPaused
from ._carry_over import AgentCarryOver

__all__ = ['AgentCarryOver', 'AgentRunPaused']
