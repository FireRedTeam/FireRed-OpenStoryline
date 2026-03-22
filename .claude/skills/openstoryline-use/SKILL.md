---
name: openstoryline-use
description: Use this skill when OpenStoryline is already installed and the user wants to start the local MCP/Web services, create or continue a session, send editing instructions, perform multi-turn re-editing, and verify rendered video outputs, as well as Chinese requests like “启动 OpenStoryline”, “把 OpenStoryline 跑起来”, “用 OpenStoryline 剪视频”.
---

# OpenStoryline Usage Skill

你负责在“已安装完成”的前提下，执行 OpenStoryline 的实际剪辑流程。
OpenStoryline 是一个剪辑 Agent，用户可使用自己的素材，通过自然语言对话的方式剪辑视频。内置素材搜索、内容理解、生成字幕、文字转语音等功能，用户可以多次提出具体的剪辑/修改意见。

目标是：使用已有脚本，稳定地完成一次从启动服务到产出视频的闭环；并且支持在同一个 `session_id` 上继续对话、二次编辑、重新生成新视频。

## Scope

此技能只处理“使用与剪辑”：

1. 检查并修改 `config.toml` 的必要字段。
2. 启动 MCP server。
3. 启动 `uvicorn agent_fastapi:app`。
4. 创建 session 并发送剪辑请求。
5. 等待并验证输出视频产物。
6. 在同一个 `session_id` 上继续对话，执行二次编辑。
7. 验证二次编辑后是否生成了新的 `output_*.mp4`。

不处理完整安装流程（依赖安装、模型下载、资源下载等），那是安装技能的范围。如果在启动时遇到问题，怀疑是安装问题，再去查看安装Skill openstoryline-install

## Core Rules

1. 默认只监听 `127.0.0.1`，不要主动暴露到局域网。
2. 优先复用现有脚本，不要重复造轮子：
   - 修改配置脚本：位于代码仓库`scripts/update_config.py`
   - Web 服务桥接脚本位于当前 skill 目录下的 `scripts/bridge_openstoryline.py`。请先定位当前 skill 目录，再拼接`scripts/bridge_openstoryline.py`  
3. 长驻服务（MCP / Web）必须按“长驻进程”方式启动，并持续观察日志；不要把启动命令当成一次性探测命令。
4. **不要**在启动命令后面追加这些包装：
   - `| head`
   - `| tail`
   - `grep`
   - `timeout`
   - `sleep`
   - `pkill`
   - 以及其它会截断日志、提前退出、强行杀进程的包装
5. 询问用户需要剪辑哪些素材及其路径。
6. 第一轮创建 session 后返回的 `session_id` 必须保存；后续继续对话、二次编辑都依赖它。
7. 如果服务端提示“上一条消息尚未完成，请稍后再发送”，不要新建 session；优先等待，必要时只终止卡住的本地 bridge 进程，然后继续复用原 `session_id` 重试。
8. 不要在任务执行中途主动终止 MCP / Web 服务，除非用户明确要求停止，或者服务本身已经确认失活。
9. 每次完成任务后，都要向用户明确返回：
    - `session_id`
    - 最终视频 `.mp4` 的完整路径
    - 如有二次编辑，还要说明是否生成了新的输出文件
10. 下面的示例命令都使用`source .venv/bin/activate`作为示例，你需要根据用户实际使用的环境，替换成正确的命令（例如`conda activate `）。
11. 遇到端口被占用的情况，优先换一个端口。

## OpenClaw Execution Strategy (Important)

如果你是OpenClaw，一定注意以下关键点：

### 长驻服务怎么跑
对于以下两类命令：

- `PYTHONPATH=src python -m open_storyline.mcp.server`
- `uvicorn agent_fastapi:app --host 127.0.0.1 --port 8005`

必须按“长驻进程”处理：

1. 用 `exec` 启动，并开启 PTY（如果工具支持 `pty: true`，就开启）。
2. 启动后不要立刻判失败，MCP Server启动需要几分钟的时间。
3. 用 `process poll` / `process log` 持续观察返回的内容，一定不要急着杀掉进程。
4. 看到成功日志再继续下一步。

### 一次性命令怎么跑
以下命令适合普通一次性 `exec`：

- 修改 `config.toml`
- 创建 session
- 在现有 `session_id` 上继续对话
- 查找 `.mp4`
- 查看文件大小

### 观察哪个日志最有用
实测中，**Web 服务日志** 最适合看剪辑进度。  
常见正常流程节点包括：

- `filter_clips`
- `group_clips`
- `generate_script`
- `generate_voiceover`
- `render_video`

如果 bridge 脚本还在等待，不代表系统没在工作；可能只是服务端还在处理。

## Standard Workflow (OpenClaw)

### 0) 确认仓库根目录

后续命令中的 `<repo-root>` 指向 OpenStoryline 仓库根目录，例如：

```bash
/Users/yourname/Desktop/code/Openstoryline/FireRed-Openstoryline
```

所有命令都默认在这个目录下执行，并先激活环境。

---

### 1) 进入项目根目录并配置

#### 必填配置
在开始剪辑前，以下 6 个字段必须有值，否则模型调用会失败。你必须先向用户询问这些字段的具体值，然后用脚本进行修改：

- `[llm].model`
- `[llm].base_url`
- `[llm].api_key`
- `[vlm].model`
- `[vlm].base_url`
- `[vlm].api_key`

直接可用命令（在仓库根目录执行，以.venv为例）：

```bash
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set llm.model=REPLACE_WITH_REAL_MODEL
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set llm.base_url=REPLACE_WITH_REAL_URL
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set llm.api_key=sk-REPLACE_WITH_REAL_KEY

cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set vlm.model=REPLACE_WITH_REAL_MODEL
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set vlm.base_url=REPLACE_WITH_REAL_URL
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set vlm.api_key=sk-REPLACE_WITH_REAL_KEY
```

#### 选填配置
以下是常见可选项，按需设置：

##### 1. MCP 端口（端口冲突时）
```bash
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set local_mcp_server.port=8002
```

##### 2. 素材检索（Pexels）
```bash
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set search_media.pexels_api_key=REPLACE_WITH_PEXELS_KEY
```

##### 3. TTS（如果需要配音）
在以下 3 家中选一家填写即可：

```bash
# minimax
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set generate_voiceover.providers.minimax.base_url=https://api.minimax.chat/v1/t2a_v2
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set generate_voiceover.providers.minimax.api_key=REPLACE_WITH_MINIMAX_KEY

# bytedance
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set generate_voiceover.providers.bytedance.uid=REPLACE_UID
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set generate_voiceover.providers.bytedance.appid=REPLACE_APPID
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set generate_voiceover.providers.bytedance.access_token=REPLACE_ACCESS_TOKEN

# 302
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set generate_voiceover.providers.302.base_url=https://REPLACE_BASE_URL
cd <repo-root> && source .venv/bin/activate && python scripts/update_config.py --config ./config.toml --set generate_voiceover.providers.302.api_key=REPLACE_API_KEY
```

---

### 2) 启动 MCP Server

注意 MCP Server 的启动可能需要几分钟，务必耐心等待，不要急着 kill 掉进程。
macOS/Linux:
```bash
cd <repo-root> && source .venv/bin/activate && PYTHONPATH=src python -m open_storyline.mcp.server
```

Windows:
```powershell
cd <repo-root>
. .venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
python -m open_storyline.mcp.server
```

看到类似以下日志，视为成功：

```text
Uvicorn running on http://127.0.0.1:8001
```

---

### 3) 启动 Web 服务（uvicorn）

macOS/Linux:
```bash
cd <repo-root> && source .venv/bin/activate && uvicorn agent_fastapi:app --host 127.0.0.1 --port 8005
```

出现以下日志即成功：

```text
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8005 (Press CTRL+C to quit)
```

---

### 4) 创建剪辑会话

注意把地址替换为上一步启动 Web 服务时的真实地址。
```bash
curl -s -X POST "http://127.0.0.1:8005/api/sessions"
```
这一步会创建一个剪辑会话，拿到一个`session_id`，**非常重要**！只要 Web 服务还活着，你就可以拿着这个`session_id`进行上传素材、多轮对话，务必保留。

### 5) 上传素材

#### 方案1：
```bash
curl -s -X POST "http://127.0.0.1:8005/api/sessions/{session_id}/media" -F "files=@/absolute/path/input.mp4"
```

#### 方案2：对于大文件，建议直接走本地copy
```bash
cp path/to/source.mp4 <repo-root>/outputs/{session_id}/media
```

### 6) 开始剪辑对话（自动创建 session）

使用 Skill 自带 bridge 脚本.
  - skills-root: 当前 skill 所在目录。
  - session-id: 填写上一步拿到的 `session_id`
  - base-url 填写 Web 服务的 url
  - prompt 用户的剪辑需求
  - lang 用户使用的语言类型，目前仅支持 zh / en，设置一次即可。

```bash
cd <repo-root> && source .venv/bin/activate && python <skills-root>/scripts/bridge_openstoryline.py \
  --session-id <session_id> \
  --base-url http://127.0.0.1:8005 \
  --prompt "剪一个小红书风格视频" \
  --lang "zh"
```

---

### 7) 等待并观察剪辑进度

有时剪辑Agent会先询问剪辑需求；有时会直接开始剪辑，剪辑可能需要几分钟，尤其是带文案、配音、渲染时。

#### 正确做法
1. 持续 `poll` 当前 bridge 脚本对应的进程会话。
2. 同时查看 Web 服务日志。剪辑 Agent 会实时更新自己的进度。
3. 只要 Web 服务日志仍在推进，就继续等待，不要随便重启服务。

#### 实战经验
如果 bridge 命令还没返回，但 Web 服务日志里已经在跑节点，这通常说明服务端仍在正常工作，不要误判为失败。

---

### 8) 第二轮：在同一个 session 上继续聊天

根据助手回复，继续向它发出剪辑要求。例如，助手制定了一个剪辑计划，请求确认。那么：

```bash
cd <repo-root> && source .venv/bin/activate && python <skills-root>/scripts/bridge_openstoryline.py \
  --base-url http://127.0.0.1:8005 \
  --session-id <上一步session_id> \
  --prompt "开始剪辑"
```

或者需要调整：

```bash
cd <repo-root> && source .venv/bin/activate && python <skills-root>/scripts/bridge_openstoryline.py \
  --base-url http://127.0.0.1:8005 \
  --session-id <上一步session_id> \
  --prompt "使用欢快的BGM"
```

---

### 9) 检查首轮产物

一般来说，剪辑agent的输出会直接写出剪辑产物的路径。
如果没有，优先检查：

```bash
cd <repo-root> && find .storyline/.server_cache/<session_id> -name "output_*.mp4" 2>/dev/null
```

#### 判定标准
有 `output_*.mp4` 即认为剪辑成功。

### 10) 发送视频

将生成的视频发送给用户观看，询问用户反馈。

#### OpenClaw + 飞书 APP 场景视频发送指南

如果你是 OpenClaw 且用户使用手机飞书 APP，使用如下专属指南。要求：
- Python 3.6+
- 已安装 `requests`
  ```bash
  python3 -m pip install requests
  ```
- OpenClaw 已配置飞书渠道

运行脚本示例如下，此脚本会自动从 ~/.openclaw/openclaw.json 读取飞书凭证。
receive-id 的选择
- oc_xxx -> chat_id：发到群聊或当前单聊会话，优先推荐
- ou_xxx -> open_id：发给指定用户 
- on_xxx -> user_id：仅当明确拿到的是 user_id 时再使用
```bash
cd <repo-root> && source .venv/bin/activate && python <skills-root>/scripts/feishu_file_sender.py --help
cd <repo-root> && source .venv/bin/activate && python <skills-root>/scripts/feishu_file_sender.py --file /absolute/path/to/video.mp4 --receive-id-type chat_id --receive-id oc_xxx
```

### 11) 二次编辑

在已经生成视频的情况下，如果用户对视频内容不满意，**可以复用同一个 `session_id` 继续修改视频**，并生成新的 `output_*.mp4`。

#### 示例：修改文案风格

```bash
cd <repo-root> && source .venv/bin/activate && python <skills-root>/scripts/bridge_openstoryline.py \
  --base-url http://127.0.0.1:8005 \
  --session-id <session_id> \
  --prompt "帮我把文案换成更欢乐、更有活力的风格"
```

- 同一个 `session_id` 下，系统会重新跑相应节点,然后重新渲染.
- 同一个 `session_id` 下会出现新的 `render_video_*` 目录。
- 新目录里会生成新的 `output_*.mp4`。

---

## Security Note

只有当用户明确要求手机 / LAN 访问时，才改为 `--host 0.0.0.0`。  
同时提醒：仅在可信网络使用，避免开放到公共网络。