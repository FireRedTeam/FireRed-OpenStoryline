# Fix local MCP media injection to avoid oversized base64 payload (fixes #11)

Hi maintainers,

Thank you for this project and for the design context you shared. This PR is based on my understanding of the codebase and the discussion in **#11**. I am not fully familiar with your internal conventions, so please treat it as a small, pragmatic suggestion rather than a final design. I am happy to adjust anything to better match your style and architecture.

---

## Problem

In the default setup, the **Agent and MCP server run on the same machine** (e.g. `connect_host = "127.0.0.1"`). The current implementation inlines **every media file** as gzip + base64 into the MCP HTTP request body for both `load_media` and dependency payloads. When there are many or large videos:

- The request body can reach **hundreds of MB**.
- That can lead to connection drops (`httpx.ReadError` / `ClientDisconnect`) and `load_media` may fail.

This matches the failure mode described in **#11**.

---

## What this PR changes (reason and result for each part)

### 1. Strategy: `should_inline_media_as_base64(server_cfg)` and config `inline_media`

| | |
|---|---|
| **Original issue** | There was no single place to choose “local vs remote” mode; base64 behavior was hard-coded in multiple places. |
| **Why change** | To have one clear switch and an **explicit config** so that Docker, LAN IP, or hostname (when still the same machine) are not misclassified as remote. |
| **After change** | Under `[local_mcp_server]`, **`inline_media`** can be `"auto"`, `"always"`, or `"never"`. **`"auto"`** (default): behavior is derived from `connect_host` — if it is `127.0.0.1`, `localhost`, `::1`, or **`0.0.0.0`** (e.g. Docker), we use path-only; otherwise we use base64. **`"always"`**: always use base64. **`"never"`**: always use path-only (e.g. when the same machine is reached via LAN IP or hostname). For typical local setups no config change is needed; Docker with `0.0.0.0` is treated as local in `"auto"`; for same-machine via LAN or hostname, set `inline_media = "never"`. |

---

### 2. `compress_payload_to_base64(payload, server_cfg)`

| | |
|---|---|
| **Original issue** | Every call converted all path-only items in the payload to base64, so dependency payloads from earlier nodes were re-encoded and the HTTP body grew again. |
| **Why change** | In local (path-only) mode, file content does not need to be sent over the wire; sending paths only is sufficient. |
| **After change** | If `should_inline_media_as_base64(server_cfg)` is `False`, the function returns the payload unchanged (no-op). Recursive handling still receives `server_cfg`. |

---

### 3. `inject_media_content_before` — `load_media` branch (path-only and path semantics)

| | |
|---|---|
| **Original issue** | For every file under `media_dir`, the code called `FileCompressor.compress_and_encode(path)` and sent `path`, `base64`, and `md5`, which made the request very large. |
| **Why change** | When the Agent and MCP share the same machine, only the path is needed; the MCP server can read the file from disk. |
| **After change** | When the strategy says path-only (no base64): we build `path` relative to **media_dir** (not cwd), and send `path`, `orig_path`, and `orig_md5: None` without base64. When the strategy says base64, behavior is unchanged (we still send base64 and md5). |

---

### 4. Path: media_dir-relative with try/except fallback

| | |
|---|---|
| **Original issue** | Using paths relative to `os.getcwd()` is fragile: if the Agent and MCP are started with different working directories (e.g. systemd, Docker, different shells), the same relative path may point to the wrong place or fail. |
| **Why change** | Using a configured **media_dir** as the base for relative paths is more stable; the server resolves paths with `server_cfg.project.media_dir`, so cwd does not matter. |
| **After change** | We compute `rel_path = path.resolve().relative_to(media_root)`. If the file is not under `media_root` (e.g. symlink), `relative_to` raises `ValueError`; we catch it and fall back to an **absolute path** so the request still works and the server’s `_load_item` can use that path as-is. |

---

### 5. `BaseNode._load_item` — path-only branch (resolve and security)

| | |
|---|---|
| **Original issue** | Only the “base64 + path” branch was implemented. When the client sent path-only (a `path` but no `base64`), `new_item` did not receive `path` / `orig_path` / `orig_md5`, so `LoadMediaNode` and downstream logic could not use the item correctly. |
| **Why change** | The server must accept path-only input, resolve it to a real file path, and set `path`, `orig_path`, and `orig_md5` so the rest of the pipeline sees a consistent structure. |
| **After change** | When `item_path` is present and `item_base64` is not: (1) If `item_path` is **absolute**, we use it as `full_path`. (2) If it is **relative**, we resolve with `media_root = Path(server_cfg.project.media_dir).resolve()` and `full_path = (media_root / item_path).resolve()`, and we require `full_path.relative_to(media_root)` to succeed so paths like `../../../etc/passwd` are rejected. We then set `new_item['path'] = str(full_path)`, `new_item['orig_path'] = str(item_path)`, and `new_item['orig_md5'] = item_md5`. |

---

### 6. `BaseNode._pack_item` — path-only return

| | |
|---|---|
| **Original issue** | After processing, the code always re-encoded the file to base64 when returning to the client (unless md5 matched and orig_path was used). So even when the client had sent path-only, the response again contained base64. |
| **Why change** | In local path-only mode, we want end-to-end path-only: the client sends a path, the server reads from disk, and the server returns the path again, without putting file content in the response. |
| **After change** | If `orig_path` is set and `orig_md5` is `None`, we treat it as path-only input and return only `item['path'] = orig_path` (no base64). Otherwise we keep the existing md5 / base64 logic. |

---

## Files changed

- **`src/open_storyline/mcp/hooks/node_interceptors.py`**  
  Strategy using **`inline_media`** (`auto` / `always` / `never`) and auto-detect (local hosts include **`0.0.0.0`** for Docker). `compress_payload_to_base64` is a no-op when path-only is used. The `load_media` branch sends path-only with a media_dir-relative path and a try/except fallback to an absolute path.

- **`src/open_storyline/config.py`**  
  `MCPConfig.inline_media`: `Literal["auto", "always", "never"]` with default `"auto"`.

- **`config.toml`**  
  Under `[local_mcp_server]`, `inline_media = "auto"` is added with a short comment for Docker/LAN use.

- **`src/open_storyline/nodes/core_nodes/base_node.py`**  
  `_load_item`: path-only branch with media_dir resolution and a path-traversal check. `_pack_item`: path-only return when `orig_md5 is None`.

---

## Impact

- **Stability**: Loading many media files should no longer cause an oversized HTTP request; path-only keeps the payload small and should help avoid `ReadError` / `ClientDisconnect`.
- **Compatibility**: Requests that send base64 are unchanged; path-only is an additional supported shape.
- **Local vs remote**: In **auto** mode, `127.0.0.1` / `localhost` / `::1` / **`0.0.0.0`** (Docker) use path-only; any other host uses base64. You can override with **`inline_media = "never"`** when the same machine is reached via LAN IP or hostname, and **`inline_media = "always"`** to force base64.

Fixes **#11**.

---

Thank you for reviewing. If anything does not match your conventions or you would prefer a different approach (e.g. how to detect local vs remote, or path semantics), I am happy to follow your guidance and iterate. I look forward to your feedback.
