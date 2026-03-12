---
name: openstoryline-install
description: Install, configure, and start FireRed-OpenStoryline from source on a local machine. Use when a user asks to set up OpenStoryline, troubleshoot installation, download required resources, fill config.toml API keys, or launch the MCP and web services, as well as Chinese requests like “安装 OpenStoryline”, “配置 OpenStoryline”, “启动 OpenStoryline”, “把 OpenStoryline 跑起来”, “修复 OpenStoryline 安装问题”, or “排查 OpenStoryline 启动失败”.
---

# OpenStoryline Install

Use this skill when the task is to install or repair a local source checkout of FireRed-OpenStoryline.

Keep the workflow deterministic:

1. Confirm the repo path and read the current README.md and config.toml.
2. Detect local prerequisites before changing anything.
3. Prefer a local `venv` install unless the user explicitly asks for Docker or `conda`.
4. Download resources only after Python dependencies succeed.
5. Validate imports and config loading before claiming success.
6. This skill assumes macOS, Linux, or WSL with a POSIX shell.

## What This Skill Covers

- Clone the GitHub repo if needed
- Create a Python environment
- Install Python dependencies
- Download `.storyline` models and `resource/` assets
- Fill `config.toml` model settings
- Start MCP and web servers
- Explain common installation/documentation gaps

## Preconditions

Check these first:

- `git`
- Python `>= 3.11`
- `ffmpeg`
- `wget`
- `unzip`

Optional:

- `docker`
- `conda`

If `ffmpeg`, `wget`, or `unzip` are missing, install them through the OS package manager before continuing.

Examples:

- macOS with Homebrew:
  ```bash
  brew install ffmpeg wget unzip
  ```

- Debian/Ubuntu:
  ```bash
  sudo apt-get update
  sudo apt-get install -y ffmpeg wget unzip
  ```

If no supported package manager or permission is available, stop and report the missing system dependency clearly.

## Interpreter selection

First prefer any interpreter that already exists and passes version checks:

1. A system Python `>= 3.11`
2. An already available conda Python `>= 3.11`
3. An already available pyenv Python `>= 3.11`, but only if basic stdlib modules work

Validate candidate interpreters before using them:

```bash
/path/to/python -c "import ssl, sqlite3, venv; print('stdlib_ok')"
```

If no supported interpreter already exists, peferr conda fallback:

```bash
conda create -y -n openstoryline-py311 python=3.11
conda run -n openstoryline-py311 python --version
conda run -n openstoryline-py311 python -m venv .venv
```

After a supported interpreter is found, always create a repo-local .venv and continue using .venv/bin/python for install, config validation, and service startup.

Do not duplicate the rest of the workflow for pyenv or conda unless the user explicitly asks to stay inside a conda environment.

## Preferred Install Path

From the repo root:

```bash
/path/to/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
bash download.sh
```

Notes:

- `download.sh` pulls both model weights and a large resource archive. It can take a long time and may resume after network drops.
- The resource download is required for a full local run, not just the Python package install.

## Configuration

Before starting the app, update config.toml.

You can use scripts/update_config.py.

At minimum, fill:

```bash
.venv/bin/python scripts/update_config.py --config ./config.toml --set llm.model=REPLACE_WITH_REAL_MODEL
.venv/bin/python scripts/update_config.py --config ./config.toml --set llm.base_url=REPLACE_WITH_REAL_URL
.venv/bin/python scripts/update_config.py --config ./config.toml --set llm.api_key=sk-REPLACE_WITH_REAL_KEY

.venv/bin/python scripts/update_config.py --config ./config.toml --set vlm.model=REPLACE_WITH_REAL_MODEL
.venv/bin/python scripts/update_config.py --config ./config.toml --set vlm.base_url=REPLACE_WITH_REAL_URL
.venv/bin/python scripts/update_config.py --config ./config.toml --set vlm.api_key=sk-REPLACE_WITH_REAL_KEY
```

Optional but common:

- `search_media.pexels_api_key` for searching media
- TTS provider keys under `generate_voiceover.providers.*` (choose one provider)


## Verification

Run these checks before saying installation is complete:

```bash
.venv/bin/pip check
PYTHONPATH=src .venv/bin/python -c "from open_storyline.config import load_settings; load_settings('config.toml'); print('config_ok')"
```

Also confirm key resources exist:

```bash
test -f .storyline/models/transnetv2-pytorch-weights.pth
test -d resource/bgms
```

## Start Services

There are two common paths. These are long-running processes. Do not wait for them to exit normally. Treat successful startup log lines or confirmed listening ports as success, and keep the services running in separate shells/sessions as needed.

Manual start:

```bash
PYTHONPATH=src .venv/bin/python -m open_storyline.mcp.server
```

In a second shell:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn agent_fastapi:app --host 127.0.0.1 --port 8005
```

## Expected Outputs

After a successful install:

- `.venv/` exists
- MCP listens on the configured local port (commonly `127.0.0.1:8001`)
- Web listens on the configured web port (commonly `127.0.0.1:8005`, though `run.sh` defaults may differ)

## Common Problems

### `download.sh` is slow or interrupted

Symptom:

- Large downloads stall or reconnect

Fix:

- Let `wget` continue; it supports resume behavior here
- Verify extracted outputs instead of trusting the progress meter

### Web/MCP server fails to bind

Symptom:

- `operation not permitted` while binding `127.0.0.1` or `0.0.0.0`

Fix:

- In agent sandboxes, request permission to open local listening ports
- Prefer `127.0.0.1` over `0.0.0.0` unless external access is required

## Response Pattern

When reporting status to the user, separate:

- what is installed
- what is still downloading
- what config is still missing
- what address the service is listening on

Do not say "installation complete" if only the Python packages are installed but the resource bundle is still missing.