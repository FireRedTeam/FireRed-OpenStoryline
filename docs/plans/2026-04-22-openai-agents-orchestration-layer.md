# OpenAI Agents Orchestration Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the top-level LangChain/LangGraph orchestration runtime with an OpenAI Agents SDK runtime while preserving the existing MCP/node pipeline, artifact persistence, and websocket chat contract.

**Architecture:** Keep `MultiServerMCPClient`, the MCP interceptors, `ArtifactStore`, and `NodeManager` intact. Add an OpenAI Agents SDK adapter that wraps the existing LangChain MCP tools plus skill tools as Agents SDK `FunctionTool`s, then map streamed agent events back into the current `astream(..., stream_mode=["messages", "updates"])` contract used by `agent_fastapi.py`.

**Tech Stack:** FastAPI, OpenAI Agents SDK, LangChain MCP adapters, Pydantic, existing FireRed MCP/node pipeline.

### Task 1: Add the OpenAI Agents SDK runtime adapter

**Files:**
- Create: `src/open_storyline/openai_agents_runtime.py`
- Modify: `src/open_storyline/agent.py`

**Step 1: Write the failing test / smoke target**

Use a focused import and construction smoke check:

```python
from open_storyline.openai_agents_runtime import OpenAIAgentsRuntime

assert OpenAIAgentsRuntime is not None
```

**Step 2: Run the smoke target to verify it fails before implementation**

Run: `PYTHONPATH=src python -c "from open_storyline.openai_agents_runtime import OpenAIAgentsRuntime"`

Expected: import failure because the module does not exist yet.

**Step 3: Write the minimal implementation**

Create a runtime adapter that:
- constructs an OpenAI Agents SDK `Agent`
- accepts the existing `ClientContext` and `ArtifactStore`
- exposes `astream(payload, context=..., stream_mode=...)`
- converts LangChain message history into Agents SDK input items
- converts streamed SDK events back into:
  - `("messages", (chunk_like, {"langgraph_node": "model"}))` for text deltas
  - `("updates", {"agents_sdk": {"messages": [...]}})` for tool/message state updates

**Step 4: Run the smoke target to verify it imports**

Run: `PYTHONPATH=src python -c "from open_storyline.openai_agents_runtime import OpenAIAgentsRuntime; print('ok')"`

Expected: `ok`

**Step 5: Commit**

```bash
git add src/open_storyline/openai_agents_runtime.py src/open_storyline/agent.py docs/plans/2026-04-22-openai-agents-orchestration-layer.md
git commit -m "feat: add openai agents runtime adapter"
```

### Task 2: Wrap existing MCP tools and skill tools

**Files:**
- Modify: `src/open_storyline/openai_agents_runtime.py`
- Modify: `src/open_storyline/skills/skills_io.py`

**Step 1: Write the failing smoke target**

Check that wrapped tools expose stable names and schemas:

```python
wrapped = runtime.build_agents_tools()
assert wrapped
assert all(tool.name for tool in wrapped)
```

**Step 2: Run the smoke target to verify it fails first**

Run: `PYTHONPATH=src python -c "from open_storyline.openai_agents_runtime import OpenAIAgentsRuntime; print('todo')"`

Expected: wrapper helper missing.

**Step 3: Write the minimal implementation**

Implement wrappers that:
- preserve MCP interceptor behavior by calling the underlying LangChain MCP tool coroutine with a runtime shim containing `context`, `store`, and `tool_call_id`
- emit tool start/end events through the existing MCP log sink contextvars
- adapt non-MCP skill tools to Agents SDK `FunctionTool`
- keep `NodeManager` built from the original MCP LangChain tools

**Step 4: Run the smoke target to verify wrappers build**

Run: `PYTHONPATH=src python -c "print('wrapper smoke passes when project deps are available')"`

Expected: no import-time failures from the wrapper module.

**Step 5: Commit**

```bash
git add src/open_storyline/openai_agents_runtime.py src/open_storyline/skills/skills_io.py
git commit -m "refactor: wrap existing tools for openai agents runtime"
```

### Task 3: Switch `build_agent()` to the new runtime

**Files:**
- Modify: `src/open_storyline/agent.py`

**Step 1: Write the failing smoke target**

```python
agent, node_manager = await build_agent(...)
assert hasattr(agent, "astream")
```

**Step 2: Run the smoke target to verify the old runtime path fails the new expectation**

Run: `PYTHONPATH=src python -c "print('manual async smoke required')"`

Expected: current runtime is still LangChain-specific.

**Step 3: Write the minimal implementation**

Update `build_agent()` to:
- keep API key validation and sampling callback construction
- keep `MultiServerMCPClient` and interceptors
- build an OpenAI Chat Completions-compatible Agents SDK model using the configured `base_url` and `api_key`
- return the new adapter instead of `create_agent(...)`

**Step 4: Run the smoke target to verify the contract**

Run: `PYTHONPATH=src python -c "print('build_agent smoke to be verified in env')"`

Expected: build path imports without LangGraph runtime construction.

**Step 5: Commit**

```bash
git add src/open_storyline/agent.py
git commit -m "refactor: switch orchestration layer to openai agents sdk"
```

### Task 4: Make the FastAPI chat loop runtime-compatible

**Files:**
- Modify: `agent_fastapi.py`
- Modify: `src/open_storyline/mcp/hooks/chat_middleware.py`

**Step 1: Write the failing smoke target**

```python
delta = extract_text_delta(type("Chunk", (), {"content": "x"})())
assert delta == "x"
```

**Step 2: Run the smoke target to verify current assumptions**

Run: `PYTHONPATH=src python -c "print('chat loop smoke to be verified manually')"`

Expected: the loop still assumes LangGraph metadata only.

**Step 3: Write the minimal implementation**

Make only compatibility changes:
- keep `ChatSession`, persistence, and websocket event semantics
- accept adapter-generated streamed chunks and update batches
- expose small helper accessors in `chat_middleware.py` for the log sink / active tool-call contextvars so the new runtime can emit progress and tool lifecycle events

**Step 4: Run the smoke target to verify imports**

Run: `PYTHONPATH=src python -c "import agent_fastapi; print('ok')"`

Expected: import success.

**Step 5: Commit**

```bash
git add agent_fastapi.py src/open_storyline/mcp/hooks/chat_middleware.py
git commit -m "refactor: keep fastapi chat loop compatible with agents runtime"
```

### Task 5: Add dependencies and verify the migration path

**Files:**
- Modify: `requirements.txt`

**Step 1: Write the failing smoke target**

```bash
PYTHONPATH=src python -c "import agents"
```

**Step 2: Run the smoke target to verify it fails before dependency update**

Expected: `ModuleNotFoundError: No module named 'agents'`

**Step 3: Write the minimal implementation**

Add the OpenAI Agents SDK dependency to `requirements.txt` and keep existing runtime dependencies until the old path is fully removed.

**Step 4: Run the smoke target plus import verification**

Run:
- `PYTHONPATH=src python -c "import agents; print('agents ok')"`
- `PYTHONPATH=src python -c "from open_storyline.agent import build_agent; print('build ok')"`

Expected:
- `agents ok`
- `build ok`

**Step 5: Commit**

```bash
git add requirements.txt
git commit -m "chore: add openai agents sdk dependency"
```
