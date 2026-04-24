# Codex FireRed Tool Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep FireRed's original MCP/node editing pipeline intact while swapping the top-level model runtime to Codex app-server models.

**Architecture:** Codex becomes the turn/thread runtime and the source of model selection/auth, but FireRed keeps its existing `MultiServerMCPClient`, registered node tools, `NodeManager`, and artifact flow. The bridge layer exposes FireRed tools to Codex as `dynamicTools`, handles `item/tool/call` roundtrips, and routes node-internal sampling callbacks through Codex as well.

**Tech Stack:** FastAPI, Codex app-server JSON-RPC, LangChain MCP adapters, OpenAI Agents runtime compatibility shims, pytest

### Task 1: Lock the bridge behavior with tests

**Files:**
- Modify: `tests/test_codex_runtime.py`
- Create: `tests/test_codex_sampling_handler.py`

**Step 1: Write the failing test**

Add a unit test that simulates `item/tool/call` from Codex and expects:
- a dynamic tool descriptor on `thread/start`
- a matching client response with `contentItems`
- `AIMessage(tool_call)` plus `ToolMessage` in the runtime updates

Add a second test that verifies node sampling requests use a Codex ephemeral thread/turn and convert MCP sampling prompts into Codex input items.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. conda run -n storyline pytest tests/test_codex_runtime.py tests/test_codex_sampling_handler.py -q`

Expected: FAIL because the runtime only supports chat-only Codex turns and there is no Codex sampling handler module.

### Task 2: Add app-server request/response support

**Files:**
- Modify: `src/open_storyline/codex/app_server_client.py`

**Step 1: Write the failing test**

Rely on the new runtime test to require:
- experimental API opt-in during `initialize`
- `dynamicTools` on `thread/start`
- server-request responses for `item/tool/call`

**Step 2: Write minimal implementation**

Add:
- `experimental_api` support in the initialize handshake
- `respond()` for JSON-RPC server requests
- `dynamic_tools` and `ephemeral` parameters on `thread_start()`

### Task 3: Bridge Codex tool calls back into FireRed tools

**Files:**
- Modify: `src/open_storyline/codex/runtime.py`

**Step 1: Write the failing test**

Use the runtime test from Task 1 as the red test.

**Step 2: Write minimal implementation**

Extend `CodexRuntime` to:
- accept FireRed/LangChain tools and the artifact store
- publish them as Codex `dynamicTools`
- handle `item/tool/call`
- invoke the underlying FireRed tool
- emit the same `tool_start/tool_end` sink events the UI already understands
- append `AIMessage`/`ToolMessage` protocol entries to updates

### Task 4: Route node-internal sampling through Codex too

**Files:**
- Create: `src/open_storyline/codex/sampling_handler.py`

**Step 1: Write the failing test**

Use the sampling callback test from Task 1 as the red test.

**Step 2: Write minimal implementation**

Create `make_codex_sampling_callback()` that:
- converts MCP sampling messages + media metadata into Codex `turn/start` inputs
- uses `ephemeral=True`
- returns `CreateMessageResult` text back to the node caller

### Task 5: Reconnect the Codex build path to the real FireRed toolchain

**Files:**
- Modify: `src/open_storyline/agent.py`
- Modify: `agent_fastapi.py`

**Step 1: Write the failing test**

Use the runtime + sampling tests to require:
- Codex path builds a populated `NodeManager`
- Codex runtime receives the existing FireRed tools

**Step 2: Write minimal implementation**

Refactor shared tool loading into a helper, then:
- keep existing `build_agent()` for OpenAI-compatible providers
- add `build_codex_agent()` for Codex
- switch `ChatSession.ensure_agent()` to use `build_codex_agent()` instead of raw `CodexRuntime(...)`

### Task 6: Verify end-to-end

**Files:**
- None

**Step 1: Run targeted tests**

Run: `PYTHONPATH=src:. conda run -n storyline pytest tests/test_model_provider_presets.py tests/test_codex_app_server_client.py tests/test_codex_runtime.py tests/test_codex_sampling_handler.py tests/test_openai_agents_runtime.py -q`

Expected: all pass

**Step 2: Run live smoke**

Build a real Codex agent and prompt it to call a FireRed tool such as `storyline_write_skills`. Verify:
- Codex returns an `AIMessage -> ToolMessage -> AIMessage` sequence
- the FireRed tool executes
- the final assistant response completes normally
