# Model Provider Presets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-class Gemini, NVIDIA, and Ollama provider presets so users can switch LLM/VLM providers from the frontend and use ready-made `config.toml` templates by only filling API keys.

**Architecture:** Introduce a single backend source of truth for model provider presets, expose it through a lightweight API schema, and let the frontend render preset selectors inside the existing custom model panels. Keep manual model/base URL override capability. Never persist LLM/VLM API keys in browser storage.

**Tech Stack:** FastAPI, existing vanilla JS frontend, OpenAI-compatible model endpoints, pytest.

### Task 1: Add provider preset schema and tests

**Files:**
- Create: `src/open_storyline/model_provider_presets.py`
- Create: `tests/test_model_provider_presets.py`

**Step 1: Write the failing test**

Verify the schema contains:
- `gemini`
- `nvidia`
- `ollama`

And that each preset includes `llm` and `vlm` entries with:
- `provider`
- `label`
- `docs_url`
- `model`
- `base_url`
- `api_key_env`

**Step 2: Run the test to verify it fails before implementation**

Run:

```bash
PYTHONPATH=src pytest tests/test_model_provider_presets.py -q
```

Expected: module/test import failure because the preset module does not exist yet.

**Step 3: Write the minimal implementation**

Create a backend helper that:
- defines the three provider presets
- returns a frontend-friendly schema for both `llm` and `vlm`
- keeps base URLs normalized and labels/document links centralized

**Step 4: Run the test again**

Expected: schema tests pass.

### Task 2: Expose preset schema to the frontend

**Files:**
- Modify: `agent_fastapi.py`

**Step 1: Write the failing test / smoke target**

Add a small smoke assertion or import target that confirms the new helper is wired and returns JSON-safe data.

**Step 2: Run the smoke target**

Expected: route/helper not yet available.

**Step 3: Write the minimal implementation**

Add a new API route such as:

```text
/api/meta/model_providers
```

Return a payload with separate `llm` and `vlm` preset lists so the frontend can render the same provider set for both panels while allowing different default model names.

**Step 4: Run the smoke target again**

Expected: route/helper imports and responds with the preset schema.

### Task 3: Add frontend preset switching

**Files:**
- Modify: `web/index.html`
- Modify: `web/static/app.js`

**Step 1: Write the failing smoke target**

Manual/UI target:
- LLM custom panel shows Gemini / NVIDIA / Ollama options
- VLM custom panel shows Gemini / NVIDIA / Ollama options
- Selecting a provider auto-fills `model` and `base_url`
- API key remains user-entered

**Step 2: Verify the current UI cannot do this**

Expected: there is no provider preset selector in the custom model UI.

**Step 3: Write the minimal implementation**

Update the frontend to:
- fetch `/api/meta/model_providers`
- render a preset dropdown in both custom model panels
- auto-fill `model` and `base_url` on provider selection
- keep fields editable after autofill
- default to the provider implied by current custom values when possible
- stop persisting LLM/VLM API key inputs in `sessionStorage`

**Step 4: Run the manual/UI smoke**

Expected: switching providers updates the fields and sending a request still uses `service_config.custom_models`.

### Task 4: Update docs with copy-paste-ready templates

**Files:**
- Modify: `docs/source/zh/api-key.md`
- Modify: `docs/source/en/api-key.md`

**Step 1: Add the three ready-made templates**

Provide copy-paste `config.toml` snippets for:
- Gemini
- NVIDIA
- Ollama / Gemma 4

Each snippet should only require users to replace API key values, with Ollama using a placeholder non-empty key like `ollama`.

**Step 2: Document frontend switching**

Explain that users can now:
- choose `Custom Model`
- pick a provider preset
- paste the API key
- switch providers freely per session

### Task 5: Verify end-to-end behavior

**Files:**
- No additional files required unless fixes are needed

**Step 1: Run automated checks**

Run:

```bash
PYTHONPATH=src pytest tests/test_model_provider_presets.py -q
PYTHONPATH=src pytest tests/test_openai_agents_runtime.py -q
PYTHONPATH=src python -c "import agent_fastapi; print('agent_fastapi import ok')"
```

**Step 2: Run frontend smoke**

Verify from the running app that:
- preset dropdowns appear
- Gemini/NVIDIA/Ollama can be switched freely
- API key fields are not restored from storage after reload

**Step 3: Commit**

```bash
git add docs/plans/2026-04-22-model-provider-presets.md \
        src/open_storyline/model_provider_presets.py \
        tests/test_model_provider_presets.py \
        agent_fastapi.py \
        web/index.html \
        web/static/app.js \
        docs/source/zh/api-key.md \
        docs/source/en/api-key.md
git commit -m "feat: add switchable model provider presets"
```
