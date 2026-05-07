---
name: add-new-node
description: 在 FireRed-OpenStoryline 中添加新节点功能的完整指南。适用于用户想要扩展新的视频编辑节点（如新的 AI 处理步骤、自定义滤镜、数据转换等）时使用。触发词：添加新节点、新建节点、扩展节点、add node、new node。
---

# FireRed-OpenStoryline 添加新节点指南

## 系统架构速览

FireRed-OpenStoryline 是一个基于 MCP（Model Context Protocol）的 AI 视频剪辑系统。整个编辑流程由一系列**节点（Node）**组成 DAG（有向无环图），每个节点是一个独立的处理单元，通过 MCP Tool 的形式暴露给 LLM Agent 调用。

### 核心文件位置

```
src/open_storyline/
├── nodes/
│   ├── core_nodes/          # 所有节点实现（每个节点一个文件）
│   │   ├── base_node.py     # 基类 BaseNode + NodeMeta
│   │   ├── load_media.py    # 示例：入口节点
│   │   ├── split_shots.py   # 示例：处理节点
│   │   └── ...
│   ├── node_schema.py       # 所有节点的 Input/Output Pydantic 模型
│   ├── node_manager.py      # 节点注册与依赖管理器
│   ├── node_state.py        # 节点执行状态（session、artifact、llm 等）
│   └── node_summary.py      # 节点执行摘要（日志/进度）
├── utils/register.py        # 全局 NODE_REGISTRY 注册表
└── config.py                # Settings 配置 Schema（Pydantic）
config.toml                  # 主配置文件
```

---

## 节点执行机制

### Mode 模式

每个节点调用时会传入 `mode` 参数：

| mode | 调用方法 | 含义 |
|------|----------|------|
| `"auto"` | `process()` | 正常执行核心逻辑（LLM 调用、算法处理等） |
| `"skip"` | `default_process()` | 跳过该节点，返回空/透传数据 |
| `"default"` | `default_process()` | 同 skip，使用默认/简化逻辑 |

### 数据流转

上游节点的输出会通过 `inputs` 字典注入到下游节点，key 为上游节点的 `node_kind`：

```python
# 在 process() 中获取上游数据
prior_data = inputs.get("split_shots", {})   # 上游 split_shots 节点的输出
clips = prior_data.get("clips", [])
```

### 节点状态对象 NodeState

```python
node_state.session_id        # 当前 session ID
node_state.artifact_id       # 当前执行的 artifact ID（唯一标识本次调用）
node_state.llm               # LLM 客户端（可用于调用语言模型）
node_state.node_summary      # 用于记录日志和进度

# 日志记录方法
node_state.node_summary.info_for_user("显示给用户看的进度信息")
node_state.node_summary.info_for_llm("给 LLM 看的执行信息")
node_state.node_summary.add_warning("警告信息", artifact_id=node_state.artifact_id)
node_state.node_summary.add_error("错误信息", artifact_id=node_state.artifact_id)
```

---

## 添加新节点的完整步骤

### Step 1：在 `node_schema.py` 定义 Input/Output Schema

打开 `src/open_storyline/nodes/node_schema.py`，参考已有模型添加：

```python
# ===== 新节点的 Input Schema =====
class MyNewNodeInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: 执行核心逻辑; skip: 跳过; default: 使用默认行为"
    )
    # 添加节点特有的参数
    my_param: str = Field(
        default="",
        description="用户对该节点的自定义参数说明"
    )

# ===== 新节点的 Output Schema（可选，用于文档和类型提示） =====
class MyNewNodeOutput(BaseModel):
    result: List[SomeType] = Field(
        default_factory=list,
        description="节点输出结果"
    )
```

**注意事项：**
- `Input` 类必须继承 `BaseInput`（已包含 `mode` 字段）或 `BaseModel`
- 字段 `description` 会直接透传为 MCP Tool 的参数说明，写清楚对 LLM 很重要
- 如果节点不需要特殊输入参数，可以直接用 `class MyNewNodeInput(BaseInput): ...`（三个点代表空 body）

---

### Step 2：创建节点实现文件

在 `src/open_storyline/nodes/core_nodes/` 下新建文件，例如 `my_new_node.py`：

```python
from typing import Any, ClassVar, Dict, List, Type
from pydantic import BaseModel

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_schema import MyNewNodeInput
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.register import NODE_REGISTRY


@NODE_REGISTRY.register()           # 必须：注册到全局注册表
class MyNewNode(BaseNode):

    # 必须：节点元数据
    meta = NodeMeta(
        name="my_new_node",         # MCP Tool 名称（snake_case，唯一）
        description=(               # MCP Tool 描述，LLM 根据此决定何时调用
            "这个节点的功能说明，写清楚它做什么、"
            "什么情况下应该调用它"
        ),
        node_id="my_new_node",      # 节点唯一 ID（通常与 name 相同）
        node_kind="my_new_node",    # 节点类型（同类替代节点共享同一 kind）
        require_prior_kind=[        # 执行 process() 时必须已完成的前置节点 kind
            "filter_clips",
        ],
        default_require_prior_kind=[  # 执行 default_process() 时的前置依赖
            "filter_clips",
        ],
        next_available_node=[       # 执行完本节点后，下游可选的节点 ID
            "plan_timeline",
        ],
        priority=5,                 # 同 kind 中多个节点时的优先级（越大越优先）
    )

    input_schema: ClassVar[Type[BaseModel]] = MyNewNodeInput

    async def default_process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        mode != "auto" 时执行。
        通常：跳过处理，透传上游数据或返回空结果。
        """
        node_state.node_summary.info_for_user(
            f"[{self.meta.node_id}] 跳过，使用默认结果"
        )
        return {"result": []}

    async def process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        mode == "auto" 时执行。核心业务逻辑在此实现。
        """
        # 1. 获取上游数据（key 为上游节点的 node_kind）
        prior_clips = inputs.get("filter_clips", {}).get("clip_captions", [])

        node_state.node_summary.info_for_user(
            f"[{self.meta.node_id}] 开始处理，共 {len(prior_clips)} 个 clip"
        )

        # 2. 你的核心处理逻辑
        result = []
        for clip in prior_clips:
            # ... 处理每个 clip
            result.append({...})

        node_state.node_summary.info_for_user(
            f"[{self.meta.node_id}] 处理完成，输出 {len(result)} 条结果"
        )

        return {"result": result}
```

**NodeMeta 字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | MCP Tool 的调用名，snake_case，全局唯一 |
| `description` | str | LLM 看到的工具描述，决定何时调用 |
| `node_id` | str | 节点唯一标识，通常与 name 相同 |
| `node_kind` | str | 节点功能类别，同类节点共享 kind（用于依赖解析） |
| `require_prior_kind` | List[str] | `process()` 的前置依赖 kind 列表 |
| `default_require_prior_kind` | List[str] | `default_process()` 的前置依赖 kind 列表 |
| `next_available_node` | List[str] | 执行完后下游可选节点的 node_id 列表 |
| `priority` | int | 同 kind 下多实现时的优先级，默认 5 |

---

### Step 3：在 `config.toml` 中注册节点

打开根目录的 `config.toml`，找到 `[local_mcp_server]` 部分，将新节点类名加入 `available_nodes`：

```toml
[local_mcp_server]
available_node_pkgs = [
    "open_storyline.nodes.core_nodes"    # 扫描这个包下的所有模块
]
available_nodes = [
    "LoadMediaNode",
    "SplitShotsNode",
    # ... 其他现有节点 ...
    "MyNewNode",                          # 新增这行
]
```

**机制说明：**
1. 系统启动时会扫描 `available_node_pkgs` 中的所有模块，触发 `@NODE_REGISTRY.register()` 自动注册
2. 然后只实例化 `available_nodes` 中列出的类，将其包装为 MCP Tool
3. 新节点文件放在 `core_nodes/` 下会被自动扫描到，不需要手动 import

---

### Step 4（可选）：添加节点独立配置

如果节点需要从 `config.toml` 读取参数（如模型路径、阈值等），在 `config.py` 中新增 Config 类：

```python
# src/open_storyline/config.py

class MyNewNodeConfig(ConfigBaseModel):
    threshold: float = Field(default=0.5, description="处理阈值")
    model_path: Path = Field(..., description="模型权重路径")

class Settings(ConfigBaseModel):
    # ... 现有字段 ...
    my_new_node: MyNewNodeConfig    # 新增这行
```

然后在 `config.toml` 添加对应 section：

```toml
[my_new_node]
threshold = 0.5
model_path = "./models/my_model.pth"
```

在节点实现中通过 `self.server_cfg` 访问：

```python
def __init__(self, server_cfg: Settings) -> None:
    super().__init__(server_cfg)
    self.threshold = self.server_cfg.my_new_node.threshold
    self.model_path = self.server_cfg.my_new_node.model_path
```

---

## 完整示例：最简单的透传节点

下面是一个完整可运行的最简节点，直接透传上游数据：

```python
# src/open_storyline/nodes/core_nodes/passthrough_node.py

from typing import Any, ClassVar, Dict, Type
from pydantic import BaseModel

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_schema import BaseInput
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.register import NODE_REGISTRY


class PassthroughNodeInput(BaseInput):
    ...  # 只继承 mode 字段，无额外参数


@NODE_REGISTRY.register()
class PassthroughNode(BaseNode):
    meta = NodeMeta(
        name="passthrough",
        description="透传节点：直接将上游数据传递给下游，不做任何处理",
        node_id="passthrough",
        node_kind="passthrough",
        require_prior_kind=["filter_clips"],
        default_require_prior_kind=["filter_clips"],
        next_available_node=["plan_timeline"],
    )
    input_schema: ClassVar[Type[BaseModel]] = PassthroughNodeInput

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self.process(node_state, inputs)

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        clips = inputs.get("filter_clips", {}).get("clip_captions", [])
        node_state.node_summary.info_for_user(f"透传 {len(clips)} 个 clip")
        return {"clip_captions": clips}
```

---

## 环境说明

本项目使用 **conda 虚拟环境**（不是 venv），环境名为 `storyline3`。所有 Python 命令需在该环境下运行：

```bash
# 验证节点可以正常导入
conda run -n storyline3 bash -c "PYTHONPATH=src python -c 'from open_storyline.nodes.core_nodes.my_new_node import MyNewNode; print(MyNewNode.meta.name)'"

# 验证完整注册流程（模拟服务启动时的扫描）
conda run -n storyline3 bash -c "PYTHONPATH=src python -c \"
from open_storyline.utils.register import NODE_REGISTRY
NODE_REGISTRY.scan_package('open_storyline.nodes.core_nodes')
print('MyNewNode in registry:', 'MyNewNode' in NODE_REGISTRY.list())
\""
```

---

## 节点调试技巧

### 开启 developer_mode

`config.toml` 中设置：
```toml
[developer]
developer_mode = true
```

开启后，节点异常会返回完整 traceback 而非简短错误信息。

### 查看节点执行历史

通过 MCP Tool `read_node_history` 可以读取任意 artifact_id 对应的执行结果：

```
read_node_history(query_artifact_id="xxx-artifact-id")
```

### LLM 调用示例

如果节点需要调用 LLM，通过 `node_state.llm` 使用：

```python
from open_storyline.mcp.sampling_requester import LLMClient

async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
    llm: LLMClient = inputs.get("llm") or node_state.llm
    
    response = await llm.ainvoke([
        {"role": "user", "content": "你的 prompt"}
    ])
    result_text = response.content
    ...
```

---

## 常见错误排查

| 错误现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 节点未出现在 MCP Tool 列表 | 类名未加入 `available_nodes` | 检查 `config.toml` |
| `KeyError: 'MyNewNode'` | 类名拼写错误或文件未被扫描到 | 检查文件在 `core_nodes/` 目录下，且类名与 config 一致 |
| `require_prior_kind` 依赖未满足 | 上游节点未执行 | 确认前置节点已执行并有输出 |
| `inputs.get("xxx")` 返回 None | node_kind 写错，或上游节点输出 key 不对 | 打印 `inputs.keys()` 检查实际 key |
| `Settings` 初始化报错 | 新增了 Config 字段但 config.toml 未对应添加 | 补充 `config.toml` 中的 section |

---

## 已有节点的节点类型（node_kind）速查

| node_kind | 节点类 | 功能 |
|-----------|--------|------|
| `load_media` | LoadMediaNode | 加载本地媒体文件 |
| `load_media` | SearchMediaNode | 从 Pexels 搜索媒体（同 kind） |
| `split_shots` | SplitShotsNode | 镜头分割（TransNetV2） |
| `asr_node` | LocalASRNode | 语音识别 |
| `speech_rough_cut` | SpeechRoughCutNode | 语音粗剪 |
| `understand_clips` | UnderstandClipsNode | VLM 视觉理解 |
| `filter_clips` | FilterClipsNode | 智能筛选 |
| `group_clips` | GroupClipsNode | 镜头分组 |
| `generate_script` | GenerateScriptNode | 脚本/字幕生成 |
| `tts` | GenerateVoiceoverNode | TTS 配音生成 |
| `tts` | VoiceCloneMinimaxNode | MiniMax 音色克隆 + TTS（与上行同 kind，可互替） |
| `select_bgm` | SelectBGMNode | 背景音乐选择 |
| `plan_timeline` | PlanTimelineProNode | 时间线规划 |
| `plan_timeline_ai_transition` | PlanTimelineAITransitionNode | AI 转场时间线 |
| `generate_ai_transition` | GenerateAITransitionNode | AI 转场生成 |
| `render_video` | RenderVideoNode | 最终渲染输出 |

---

## 特殊主题：节点接收音频文件输入

### 音频文件在系统中的流转路径

```
用户上传音频（Web UI）
  → 存入 session media_dir（绝对路径）
  → scan_media_dir 返回统计 + 绝对路径列表（注入到 Agent 的 system message）
  → Agent 知道有哪些音频文件可用
  → Agent 在 tool call 参数中填写 clone_audio=[{"path": "/abs/path/audio.mp3"}]
  → 拦截器 inject_media_content_before 检测到 voice_clone_minimax 且 clone_audio 未传时
      → 自动从 media_dir 扫描音频文件并注入 clone_audio
  → BaseNode.load_inputs_from_client 处理 clone_audio 字段（base64/path 双模式）
  → node.process() 中 clone_audio_list[0]["path"] 是 server 本地可读路径
```

### 为什么音频文件不经过 load_media 节点？

`load_media` 节点只处理视频和图片（VIDEO_EXTS / IMAGE_EXTS），音频文件会被 skip。
音频文件不需要 `load_media` 处理，因为：
- 视频/图片需要元数据提取（duration、fps、width×height）
- 音频只是一个文件路径，直接传给 TTS/音色克隆 API 即可

### 如果你的新节点也需要接收音频输入

1. **Input Schema** 中声明 `List[Dict[str, Any]]` 字段：

```python
class MyAudioNodeInput(BaseInput):
    audio_files: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="音频文件列表，格式：[{'path': '/abs/path/audio.mp3'}]"
    )
```

2. **拦截器自动注入**（可选）：如果希望系统自动从 media_dir 找音频注入，
   在 `node_interceptors.py` 的 `inject_media_content_before` 中参考
   `voice_clone_minimax` 的处理方式，添加 `node_id == 'my_audio_node'` 分支。

3. **节点 process() 中读取**：`BaseNode.load_inputs_from_client` 自动处理 `List[dict]`
   字段的 base64/path 双模式，process() 里直接 `inputs.get("audio_files")` 即可，
   `item["path"]` 已经是 server 本地可 open() 的路径。

### scan_media_dir 的行为

`src/open_storyline/utils/media_handler.py` 中的 `scan_media_dir` 会统计：
- 图片数量
- 视频数量
- 音频文件的绝对路径列表（`.mp3`/`.wav`/`.m4a`）

这些统计信息会注入到 Agent 的 system message，让 Agent 知道用户上传了哪些媒体。
