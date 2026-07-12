# Durable Execution with DBOS

[DBOS](https://www.dbos.dev/) is a lightweight [durable execution](https://docs.dbos.dev/architecture) library natively integrated with Pydantic AI.

## Durable Execution

DBOS workflows make your program **durable** by checkpointing its state in a database. If your program ever fails, when it restarts all your workflows will automatically resume from the last completed step.

* **Workflows** must be deterministic and generally cannot include I/O.
* **Steps** may perform I/O (network, disk, API calls). If a step fails, it restarts from the beginning.

Every workflow input and step output is durably stored in the system database. When workflow execution fails, whether from crashes, network issues, or server restarts, DBOS leverages these checkpoints to recover workflows from their last completed step.

DBOS **queues** provide durable, database-backed alternatives to systems like Celery or BullMQ, supporting features such as concurrency limits, rate limits, timeouts, and prioritization. See the [DBOS docs](https://docs.dbos.dev/architecture) for details.

The diagram below shows the overall architecture of an agentic application in DBOS.
DBOS runs fully in-process as a library. Functions remain normal Python functions but are checkpointed into a database (Postgres or SQLite).

```text
                    Clients
            (HTTP, RPC, Kafka, etc.)
                        |
                        v
+------------------------------------------------------+
|               Application Servers                    |
|                                                      |
|   +----------------------------------------------+   |
|   |        Pydantic AI + DBOS Libraries          |   |
|   |                                              |   |
|   |  [ Workflows (Agent Run Loop) ]              |   |
|   |  [ Steps (Tool, MCP, Model) ]                |   |
|   |  [ Queues ]   [ Cron Jobs ]   [ Messaging ]  |   |
|   +----------------------------------------------+   |
|                                                      |
+------------------------------------------------------+
                        |
                        v
+------------------------------------------------------+
|                      Database                        |
|   (Stores workflow and step state, schedules tasks)  |
+------------------------------------------------------+
```

See the [DBOS documentation](https://docs.dbos.dev/architecture) for more information.

## Durable Agent

Add durable execution to any [`Agent`][pydantic_ai.agent.Agent] by attaching the [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] [capability](../capabilities.md). When the agent runs inside a DBOS workflow, the capability routes [model requests](../models/overview.md) and [MCP communication](../mcp/client.md) through DBOS steps. To make a run durable, call `agent.run()` inside a `@DBOS.workflow`.

The agent stays a normal `Agent` everywhere — outside a DBOS workflow the capability is transparent, and the original agent, model, and MCP server can still be used as normal.

Custom tool functions and event stream handlers are **not automatically wrapped** by DBOS.
If they involve non-deterministic behavior or perform I/O, you should explicitly decorate them with `@DBOS.step`.

Here is a simple but complete example of attaching durable execution to an agent. All it requires is to install Pydantic AI with the DBOS [open-source library](https://github.com/dbos-inc/dbos-transact-py):

```bash
pip/uv-add pydantic-ai[dbos]
```

Or if you're using the slim package, you can install it with the `dbos` optional group:

```bash
pip/uv-add pydantic-ai-slim[dbos]
```

```python {title="dbos_durability.py" test="skip"}
from dbos import DBOS, DBOSConfig

from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSDurability

dbos_config: DBOSConfig = {
    'name': 'pydantic_dbos_agent',
    'system_database_url': 'sqlite:///dbostest.sqlite',  # (1)!
}
DBOS(config=dbos_config)

agent = Agent(
    'openai:gpt-5.2',
    instructions="You're an expert in geography.",
    name='geography',  # (2)!
    capabilities=[DBOSDurability()],  # (3)!
)


@DBOS.workflow()  # (4)!
async def answer(question: str) -> str:
    result = await agent.run(question)
    return result.output


async def main():
    DBOS.launch()
    answer_text = await answer('What is the capital of Mexico?')
    print(answer_text)
    #> Mexico City (Ciudad de México, CDMX)
```

1. This example uses SQLite. Postgres is recommended for production.
2. The agent's `name` is used to uniquely identify its workflows.
3. Attach durability via `capabilities=[...]`. The capability routes model requests and MCP communication through DBOS steps when the agent runs inside a workflow. Because DBOS workflows must be registered before `DBOS.launch()`, the agent must also be constructed before calling `DBOS.launch()`.
4. Wrap `agent.run()` in your own `@DBOS.workflow` to make the run durable.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

Because the same agent works inside and outside a DBOS workflow, [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] composes with all other [capabilities](../capabilities.md) without each needing a DBOS-specific wrapper variant.

For more information on how to use DBOS in Python applications, see their [Python SDK guide](https://docs.dbos.dev/python/programming-guide).

### Wrapper-agent path (deprecated)

!!! warning "Deprecated"
    [`DBOSAgent`][pydantic_ai.durable_exec.dbos.DBOSAgent] is the original wrapper-agent path for DBOS integration and will be removed in v3. New code should use the [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] capability shown above.

Any agent can be wrapped in a [`DBOSAgent`][pydantic_ai.durable_exec.dbos.DBOSAgent] to get a durable agent variant. `DBOSAgent` wraps `Agent.run` / `Agent.run_sync` as DBOS workflows automatically and routes model requests and MCP communication through DBOS steps. [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] also routes model requests and MCP communication through steps, but does not wrap the run: call `agent.run()` inside your own `@DBOS.workflow`.

When migrating from `DBOSAgent` to `DBOSDurability`, wrap the `agent.run()` call in a `@DBOS.workflow`.

```python {title="dbos_agent.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent

agent = Agent('openai:gpt-5.2', name='geography')
dbos_agent = DBOSAgent(agent)  # Use `dbos_agent` in place of `agent`.
```

## DBOS Integration Considerations

When using DBOS with Pydantic AI agents, there are a few important considerations to ensure workflows and toolsets behave correctly.

### Agent and Toolset Requirements

Each agent instance must have a unique `name` so DBOS can correctly resume workflows after a failure or restart.

Each [`MCPToolset`][pydantic_ai.mcp.MCPToolset] must have a unique [`id`][pydantic_ai.toolsets.AbstractToolset.id], as DBOS derives its step names and per-run tool-defs cache key from it. This field is normally optional, but is required when using DBOS. It should not be changed once the durable agent has been deployed to production, as this would break active workflows.

Tools and event stream handlers are not automatically wrapped by DBOS. You can decide how to integrate them:

* Decorate with `@DBOS.step` if the function involves non-determinism or I/O.
* Skip the decorator if durability isn't needed, so you avoid the extra DB checkpoint write.
* If the function needs to enqueue tasks or invoke other DBOS workflows, run it inside the agent's main workflow (not as a step).

Other than that, any agent and toolset will just work!

### Agent Run Context and Dependencies

DBOS checkpoints workflow inputs/outputs and step outputs into a database using [`pickle`](https://docs.python.org/3/library/pickle.html). This means you need to make sure the [dependencies](../dependencies.md) object provided to [`Agent.run()`][pydantic_ai.agent.Agent.run] / [`Agent.run_sync()`][pydantic_ai.agent.Agent.run_sync] (or `DBOSAgent.run` / `DBOSAgent.run_sync` on the deprecated path), and tool outputs can be serialized using pickle. You may also want to keep the inputs and outputs small (under \~2 MB). PostgreSQL and SQLite support up to 1 GB per field, but large objects may impact performance.

### Model Selection at Runtime

[`Agent.run(model=...)`][pydantic_ai.agent.Agent.run] supports both model strings (like `'openai:gpt-5.2'`) and model instances. A model instance can't be serialized across the step boundary, so it's sent as its `model_id` string and rebuilt inside the step. That faithfully reproduces model-name strings and models with standard providers, but not an instance whose exact behavior depends on a custom provider, client, or settings — pre-register those by passing a `models` dict to [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] and reference them by key (or pass the registered instance). The agent's own model, set at construction, is always available as the default.

To customize how a model string is built — a custom provider, or per-user credentials carried on the run's `deps` — add a [`ResolveModelId`](../capabilities.md#resolvemodelid) capability before `DBOSDurability`: it gets first crack at every string, and the resolver runs again inside the step with the run's actual `deps`, so it must be deterministic for a given `(model_id, deps)` and must not perform external I/O.

### Streaming

Because DBOS cannot stream output directly to the workflow or step call site, [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream] and [`Agent.run_stream_events()`][pydantic_ai.agent.Agent.run_stream_events] are not supported when running inside of a DBOS workflow.

For handlers with I/O side effects, pass `event_stream_handler=` to [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability]. Model events are delivered live inside each model-request step, while each tool event is delivered in its own event-handler step. As with any DBOS step, a handler may run more than once if the workflow recovers before its step is checkpointed, so keep its side effects idempotent.

Alternatively, register [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] and use [`Agent.run()`][pydantic_ai.agent.Agent.run]. Its handler runs in workflow code and must be deterministic because it re-runs on workflow replay. Tool and final-output events arrive live, while the real captured model events are replayed after each model request completes. For examples, see the [streaming docs](../agent.md#streaming-all-events).

A per-run handler passed to `Agent.run(event_stream_handler=...)` also runs workflow-side against replayed model events.

Because the model stream is consumed inside the step, cancelling it from the workflow side (e.g. with [`AgentStream.cancel()`][pydantic_ai.result.AgentStream.cancel]) is not available across the durable boundary.

### Suspended Turns and Background Mode

When a provider pauses a model turn mid-flight (Anthropic `pause_turn`) or runs it as a server-side job that's polled until it's ready ([OpenAI background mode](../models/openai.md#background-mode)), each segment runs in a separate model request step. The suspended [`ModelResponse`][pydantic_ai.messages.ModelResponse] and background job ID are checkpointed between segments, while the final response is merged and usage is recorded once. A [`message_history`](../message-history.md) ending in a suspended response is passed to the first step. Size step timeouts for one provider round trip. If an error abandons a suspended job, its provider teardown runs in a dedicated cancellation step.

### Parallel Tool Execution

Under DBOS, tools are executed in parallel by default to minimize latency. To guarantee deterministic replay and reliable recovery, DBOS waits for all parallel tool calls to complete before emitting events **in order**.
It's equivalent to the behavior of [`with agent.parallel_tool_call_execution_mode('parallel_ordered_events')`][pydantic_ai.agent.AbstractAgent.parallel_tool_call_execution_mode].

If you prefer strict ordering, you can configure the agent to run tools sequentially by setting `parallel_execution_mode='sequential'` on [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] (or on the deprecated [`DBOSAgent`][pydantic_ai.durable_exec.dbos.DBOSAgent]).

### Nested Agent Runs

A tool that calls [`Agent.run()`][pydantic_ai.agent.Agent.run] on another agent (a sub-agent, or "delegation" pattern) runs it inline by default, since DBOS already runs function tools inline: as long as the sub-agent carries its own [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] capability, its own model and tool calls are still checkpointed as DBOS steps -- they're just recorded as part of the *caller's* workflow (see `_install_workflow_wrappers`: an `agent.run()` called from inside an already-active DBOS workflow flattens into it rather than starting a nested one). That's often fine, but it means the sub-agent's run has no independent identity: it can't be recovered, retried, or inspected separately from the caller's own workflow.

To give a delegation tool's nested run its own DBOS workflow, tag it with `metadata={'nested_agent_run': True}`, declaring the engine-neutral fact that its body runs another agent. Under DBOS, this makes the tool run as its own workflow: `DBOS.workflow_id` changes for the duration of the call, so the sub-agent's own `agent.run()` (still flattening into *this* new workflow, by the same rule as above) shows up as a distinct, independently recoverable unit instead of merging into the top-level run.

```python {title="dbos_nested_agent.py" test="skip" lint="skip"}
@toolset.tool(metadata={'nested_agent_run': True})
async def delegate_to_specialist(ctx: RunContext[Deps], task: str) -> str:
    result = await specialist_agent.run(task, deps=ctx.deps)
    return result.output
```

For this to work:

- The sub-agent (`specialist_agent` above) should carry its own [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] capability, bound at agent-construction time -- otherwise its own model requests aren't checkpointed as DBOS steps.
- The tool's toolset needs a unique [`id`][pydantic_ai.toolsets.AbstractToolset.id] (same requirement as [`MCPToolset`][pydantic_ai.mcp.MCPToolset] above): without one, `DBOSDurability` can't derive a stable workflow name for it and raises a `UserError` at agent construction.
- The tool must be `async` (DBOS runs non-async tools inline in a thread, which isn't supported for workflow code).

`nested_agent_run` picks DBOS's default workflow config. For control over `max_recovery_attempts`, set `metadata={'dbos_workflow': DBOSWorkflowConfig(max_recovery_attempts=...)}` explicitly instead -- it takes precedence over a `nested_agent_run` hint on the same tool. This split matters for toolset authors: a toolset that isn't DBOS-specific (like a sub-agent/delegation toolset meant to work under any durability engine, or none) should only ever set `nested_agent_run`, leaving the DBOS-specific decision to whoever configures that particular agent's durability -- the same tag [Temporal's `TemporalDurability`](temporal.md#nested-agent-runs) and [Prefect's `PrefectDurability`](prefect.md#nested-agent-runs) read to make their own engine-specific choice.

A `FunctionToolset` with no tagged tool is never wrapped: this feature adds no requirements (no `id`, no behavior change) for tools that don't use it.

### Toolsets at Runtime

Additional toolsets can be passed per run via `agent.run(toolsets=...)` (on both the [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] and `DBOSAgent` paths). Non-executing toolsets like [`ExternalToolset`][pydantic_ai.toolsets.ExternalToolset], and [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset]s whose tools DBOS runs inline, are supported. [`MCPToolset`][pydantic_ai.mcp.MCPToolset]s and dynamic toolsets must be set when constructing the agent so their steps are registered before the workflow runs; passing them at runtime raises a `UserError`. A tool tagged `nested_agent_run` on a toolset added at runtime doesn't get its own workflow either: the tag is only discovered at agent-construction time.


## Step Configuration

You can customize DBOS step behavior, such as retries, by passing [`StepConfig`][pydantic_ai.durable_exec.dbos.StepConfig] objects to the [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] (or deprecated [`DBOSAgent`][pydantic_ai.durable_exec.dbos.DBOSAgent]) constructor:

- `mcp_step_config`: The DBOS step config to use for MCP server communication. No retries if omitted.
- `model_step_config`: The DBOS step config to use for model request steps. No retries if omitted.

For custom tools, you can annotate them directly with [`@DBOS.step`](https://docs.dbos.dev/python/reference/decorators#step) or [`@DBOS.workflow`](https://docs.dbos.dev/python/reference/decorators#workflow) decorators as needed. These decorators have no effect outside DBOS workflows, so tools remain usable in non-DBOS agents.


## Step Retries

On top of the automatic retries for request failures that DBOS will perform, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause the request to be retried more often than expected, with improper `Retry-After` handling.

When using DBOS, it's recommended to not use [HTTP Request Retries](../retries.md) and to turn off your provider API client's own retry logic, for example by setting `max_retries=0` on a [custom `OpenAIProvider` API client](../models/openai.md#custom-openai-client).

You can customize DBOS's retry policy using [step configuration](#step-configuration).

## Observability with Logfire

DBOS can be configured to generate OpenTelemetry spans for each workflow and step execution, and Pydantic AI emits spans for each agent run, model request, and tool invocation. You can send these spans to [Pydantic Logfire](../logfire.md) to get a full, end-to-end view of what's happening in your application.

For more information about DBOS logging and tracing, please see the [DBOS docs](https://docs.dbos.dev/python/tutorials/logging-and-tracing) for details.
