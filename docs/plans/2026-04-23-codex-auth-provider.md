# Codex Auth Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add official Codex app-server authentication and a Codex-backed multimodal chat runtime so FireRed can use `Codex (ChatGPT)` as both the default LLM and VLM after login.

**Architecture:** Keep the existing OpenAI-compatible provider flow for `OpenAI API`, `Gemini`, `NVIDIA`, and `Ollama`. Add a separate `codex` provider that does not use `api_key/base_url`; instead it talks to `codex app-server` over stdio JSON-RPC for account state, rate limits, and text/image turns. In the FastAPI websocket route, branch to a Codex runtime path only when `llm_model_key` or `vlm_model_key` is `__provider__:codex`.

**Tech Stack:** FastAPI, existing websocket chat loop, Codex CLI `app-server`, asyncio subprocess/JSON-RPC, existing frontend sidebar app, pytest.

### Task 1: Define Codex config and provider metadata

**Files:**
- Modify: `src/open_storyline/config.py`
- Modify: `src/open_storyline/model_provider_presets.py`
- Modify: `config.toml`
- Test: `tests/test_model_provider_presets.py`

**Step 1: Write the failing tests**

Add tests asserting:
- `build_model_provider_ui_schema()` includes `codex` in both `llm` and `vlm`
- the Gemini default becomes `gemma-4-26b-a4b-it`
- Codex presets expose `auth_kind="codex"` and do not require API key/base URL fields

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_model_provider_presets.py -q`

Expected: failure because `codex` is missing and Gemini still points to the previous model.

**Step 3: Write the minimal implementation**

- Add a `CodexConfig` model to `config.py`
- Add `codex: CodexConfig` to `Settings`
- Add `[codex]` defaults in `config.toml`
- Extend preset schema with `codex` entries for both `llm` and `vlm`
- Mark Codex presets so the frontend can identify them as auth-backed rather than API-key-backed

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_model_provider_presets.py -q`

Expected: all preset/config tests pass.

### Task 2: Add a Codex app-server client and backend API endpoints

**Files:**
- Create: `src/open_storyline/codex/app_server_client.py`
- Modify: `agent_fastapi.py`
- Test: `tests/test_codex_app_server_client.py`

**Step 1: Write the failing tests**

Add focused tests for:
- parsing device-code and browser login responses
- normalizing account/rate-limit responses
- `GET /api/codex/account` returning the transformed payload from a stub client
- `POST /api/codex/login/start` accepting `device_code` and `browser`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_codex_app_server_client.py -q`

Expected: import/module failures because the client and endpoints do not exist yet.

**Step 3: Write the minimal implementation**

- Wrap `codex app-server --listen stdio://` with an asyncio JSON-RPC client
- Support `initialize`, `account/read`, `account/login/start`, `account/login/cancel`, `account/logout`, and `account/rateLimits/read`
- Add FastAPI endpoints:
  - `GET /api/codex/account`
  - `POST /api/codex/login/start`
  - `POST /api/codex/login/cancel`
  - `POST /api/codex/logout`
  - `GET /api/codex/rate_limits`
- Keep auth state separate from provider API-key config

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_codex_app_server_client.py -q`

Expected: endpoint/client tests pass.

### Task 3: Add a Codex multimodal turn runtime

**Files:**
- Create: `src/open_storyline/codex/runtime.py`
- Modify: `agent_fastapi.py`
- Test: `tests/test_codex_runtime.py`

**Step 1: Write the failing tests**

Add tests for:
- converting prompt + image attachment into Codex input items
- streaming agent deltas into websocket-friendly assistant events
- final assistant text reconstruction from app-server notifications

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_codex_runtime.py -q`

Expected: failures because the runtime adapter does not exist.

**Step 3: Write the minimal implementation**

- Create a Codex runtime adapter that:
  - starts a thread
  - sends `turn/start`
  - converts pending media images into `localImage` items
  - streams `item/agentMessage/delta` text back to the existing websocket sender
- In `agent_fastapi.py`, bypass `ensure_agent()/astream()` when the selected provider is `codex`
- Keep the current OpenAI Agents SDK runtime untouched for the non-Codex providers

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_codex_runtime.py -q`

Expected: Codex runtime tests pass.

### Task 4: Update the frontend for Codex auth and default provider switching

**Files:**
- Modify: `web/index.html`
- Modify: `web/static/app.js`
- Test: manual browser verification plus `node --check web/static/app.js`

**Step 1: Write the failing target**

Define UI expectations:
- a `Codex Auth` card exists
- login success switches both `LLM` and `VLM` to `Codex`
- the sidebar surfaces Codex account email, plan, and rate-limit status

**Step 2: Verify the current UI does not provide this behavior**

Run: refresh the app at `http://192.168.8.191:7861/`

Expected: no Codex auth card and no provider-switching behavior.

**Step 3: Write the minimal implementation**

- Add a `Codex Auth` sidebar panel
- Add buttons for device-code login, browser login, refresh, and logout
- Show account summary and rate limits
- Include `codex` in both model selectors
- When login succeeds, auto-select `Codex` for both LLM and VLM
- Keep the custom model inputs for provider-backed presets, but avoid asking for API keys in the browser for Codex

**Step 4: Verify the updated UI**

Run:
- `node --check web/static/app.js`
- refresh the browser and validate the new sidebar controls

Expected: script parses, new UI renders, and provider switching behaves correctly.

### Task 5: End-to-end verification against the live app

**Files:**
- Modify as needed from earlier tasks

**Step 1: Run the targeted Python tests**

Run:
- `PYTHONPATH=src pytest tests/test_model_provider_presets.py -q`
- `PYTHONPATH=src pytest tests/test_codex_app_server_client.py -q`
- `PYTHONPATH=src pytest tests/test_codex_runtime.py -q`

**Step 2: Run import/smoke checks**

Run:
- `PYTHONPATH=src python -c "import agent_fastapi; print('agent_fastapi ok')"`
- `PYTHONPATH=src python -c "from open_storyline.codex.app_server_client import CodexAppServerClient; print('codex client ok')"`

**Step 3: Restart the app if required and do a live check**

Validate:
- login state is readable from `/api/codex/account`
- the sidebar switches to Codex after login
- a simple text turn works
- a simple image turn works

**Step 4: Commit**

```bash
git add agent_fastapi.py config.toml docs/plans/2026-04-23-codex-auth-provider.md src/open_storyline/config.py src/open_storyline/model_provider_presets.py src/open_storyline/codex web/index.html web/static/app.js tests/test_model_provider_presets.py tests/test_codex_app_server_client.py tests/test_codex_runtime.py
git commit -m "feat: add codex auth provider"
```
