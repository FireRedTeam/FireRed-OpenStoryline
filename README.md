<div align="center">
  <a href="#gh-light-mode-only">
    <img
      src="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/web/static/brand_white.png"
      alt="openstoryline"
      width="70%"
    />
  </a>

  <a href="#gh-dark-mode-only">
    <img
      src="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/web/static/brand_black.png"
      alt="openstoryline"
      width="70%"
    />
  </a>
  
  <p>
    <a href="./README_zh.md">🇨🇳 简体中文</a> | 
    <a href="./README.md">🌏 English</a>
  </p>
  <p>
    <a href="https://huggingface.co/FireRedTeam" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FireRedTeam-ffc107?color=ffc107&logoColor=white" style="display: inline-block;"/></a>
    <a href="https://www.modelscope.cn/studios/FireRedTeam/FireRed-OpenStoryline" target="_blank">
        <img alt="ModelScope Demo" src="https://img.shields.io/badge/ModelScope-Demo-4B6CFF?style=flat&logo=modelscope&logoColor=white" style="display: inline-block;"/></a>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
    <a href="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/docs/media/others/group_20260329.jpg"><img src="https://img.shields.io/badge/Xiaohongshu-Group-E9DBFC?style=flat&logo=xiaohongshu&logoColor=white" alt="xiaohongshu"></a>
    <a href="https://hellogithub.com/repository/FireRedTeam/FireRed-OpenStoryline" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=fb457793a87e4bdab9c4f5ca26451a7f&claim_uid=4QpcNTrHEAg3O8n&theme=small" alt="Featured｜HelloGitHub" /></a>
  </p>
</div>

<div align="center">

[🤗 HuggingFace Demo](https://fireredteam-firered-openstoryline.hf.space/) • [🌐 Homepage](https://fireredteam.github.io/demos/firered_openstoryline/)

</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/9116767e-bcd9-417a-93d8-2db4d3d5df8e" width="70%" poster=""> </video>
</div>

**FireRed-OpenStoryline** turns complex video creation into natural, intuitive conversations. Designed with both accessibility and enterprise-grade reliability in mind, FireRed-OpenStoryline makes video creation easy and friendly to beginners and creative enthusiasts alike.
> Deriving from the saying "A single spark can start a prairie fire", the name FireRed represents our vision: to spread our SOTA capabilities—honed in real-world scenarios—like sparks across the wilderness, igniting the imagination of developers worldwide to reshape the future of AI together.

## ✨ Key Features
- 🌐 **Smart Media Search & Organization**: Automatically searches online and downloads images and video clips that match your requirements. Performs clip segmentation and content understanding based on your thematic media.
- ✍️ **Intelligent Script Generation**: Combines user themes, visual understanding, and emotion recognition to automatically construct storylines and context-aware narration. Features built-in Few-shot style transfer capabilities, allowing users to define specific copy styles (e.g., product reviews, casual vlogs) via reference text, achieving precise replication of tone, rhythm, and sentence structure.
- 🎵 **Intelligent Music, Voiceover & Font Recommendations**: Supports personal playlist imports and auto-recommends BGM based on content and mood, featuring smart beat-syncing. Simply describe the desired tone—e.g., "Restrained," "Emotional," or "Documentary-style"—and the system matches suitable voiceovers and fonts to ensure a cohesive aesthetic.
- 💬 **Conversational Refinement**: Rapidly cut, swap, or resequence clips. Edit scripts and fine-tune visual details—including color, font, stroke, and position. All edits are performed exclusively via natural language prompts with immediate results.
- ⚡**Editing Skill Archiving**: Save your complete editing workflow as a custom Skill. Simply swap the media and apply the corresponding Skill to instantly replicate the style, enabling efficient batch creation.

## NEWS

* 🎬 **2026-03-30**: Added the **AI Transition Generation** feature, which automatically creates transition shots based on the ending frame of one clip, the opening frame of the next, and a natural-language description, making scene transitions smoother and the narrative more coherent.
* 🚀 **2026-03-22**: Introduced an **ASR-based rough cut skill for speech videos**, enabling automatic removal of filler words, disfluencies, and repeated sentences, with timestamp-aligned segmentation for cleaner and more efficient speech editing workflows.
* 🔥 **2026-03-12**: Integrated with **OpenClaw**, adding two OpenClaw Skills — `openstoryline-install` and `openstoryline-use` — covering the initial installation/first-run workflow and the actual usage workflow, respectively. Also added Skill usage instructions for **Claude Code**, making it easier for **Claude Code** to install and invoke the project in accordance with the repository guidelines.
* **2026-02-10**: FireRed-OpenStoryline was officially open-sourced.

> <sub>
> ⚠️ Note: AI transitions rely on third-party AIGC video generation services, and <b>the cost is relatively high</b>. Due to variations in source material quality, prompts, and model performance, the generated results are somewhat unpredictable. It is recommended to enable this feature only when needed.
> </sub>

## 🏗️ Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/FireRedTeam/fireredteam.github.io/main/demos/firered_openstoryline/pics/structure.jpg" alt="openstoryline architecture" width="800">
</p>

## ✨ Demo
<table align="center">
  <tr>
    <td align="center"><b>Zhongcao Style</b></td>
    <td align="center"><b>Humorous Style</b></td>
    <td align="center"><b>Product Picks</b></td>
    <td align="center"><b>Artistic Style</b></td>
  </tr>
  <tr>
    <td align="center"><video src="https://github.com/user-attachments/assets/28043813-1fda-4077-80d4-c6f540d7c7cb" width="220" /></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/a1e33da2-a799-4398-a1bb-b25bb5143d7c" width="220" /></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/444fd0fb-8824-4c25-b449-9309b0fcfd85" width="220" /></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/2e69fa0d-b693-4d4f-b4d2-45146254f9e8" width="220" /></td>
  </tr>

  <tr>
    <td align="center"><b>Unboxing</b></td>
    <td align="center"><b>Talking Pet</b></td>
    <td align="center"><b>Travel Vlog</b></td>
    <td align="center"><b>Year-in-Review</b></td>
  </tr>
  <tr>
    <td align="center"><video src="https://github.com/user-attachments/assets/ff1d669b-1d27-4cf8-b0be-1b141c717466" width="220" /></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/063608bb-7fbd-4841-a08f-032ae459499f" width="220" /></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/bc441dfa-e995-4575-8401-ecefa269e57b" width="220" /></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/533ef5c3-bb76-4416-bff7-825e88b00b7d" width="220" /></td>
  </tr>
</table>

> <sub>
> 🎨 <b>Effects Note:</b> Due to licensing restrictions on open-source assets, the elements (fonts/music) in the first row represent only basic effects. We <b>highly recommend</b> following the <a href="https://github.com/FireRedTeam/FireRed-OpenStoryline/blob/main/docs/source/zh/guide.md#2-%E9%AB%98%E7%BA%A7%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B">Custom Asset Library Tutorial</a> to unlock commercial-grade fonts, music, and VFX for significantly better video quality.<br>
> ⚠️ <b>Quality Note:</b> To save space in the README, the demo videos are heavily compressed. The actual output retains the original resolution by default and supports custom dimensions.<br>
> In the Demo: The <b>first row</b> shows default open-source assets (Restricted Mode); the <b>second row</b> shows Xiaohongshu App "AI Clip" asset library effects. 👉 <a href="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/docs/media/others/ai_cut_guide.png">Click to view tutorial</a><br>
> ⚖️ <b>Disclaimer:</b> User footage and brand logos shown in the demos are for technical demonstration purposes only. Ownership belongs to the original creators. Please contact us for copyright concerns.
> </sub>

## 🤖 Use Through an Agent

FireRed-OpenStoryline supports usage through Agent Skills.
We provide two Skills:

* `openstoryline-install`: for installation, configuration, and first-run verification.
* `openstoryline-use`: for starting the service and running the actual video editing workflow.

### OpenClaw

Just tell OpenClaw: “I want to try OpenStoryline. Help me install the required Skills,” and it will automatically trigger the installation.
If the installation runs into problems, use the following commands to install them manually:

```bash
openclaw skills install openstoryline-install
openclaw skills install openstoryline-use
```

If your current OpenClaw version does not support `openclaw skills install`, or if installation still fails, you can use ClawHub instead:

```bash
npx clawhub install openstoryline-install
npx clawhub install openstoryline-use
```

Once installed, you only need to send your media assets to OpenClaw, and it can help you complete the entire process from installing FireRed-OpenStoryline to generating the final video.

### Claude Code

This repository comes with built-in Claude Code Skills.
If you start Claude Code from the **root directory of this repository**, you can use the project-level Skills included in the repo directly. Claude Code can then help you install and use FireRed-OpenStoryline.

```bash
/openstoryline-install
/openstoryline-use
```

If you want to install these two Skills into your own global Claude Code configuration, run:

```bash
mkdir -p ~/.claude/skills
cp -R .claude/skills/openstoryline-install ~/.claude/skills/
cp -R .claude/skills/openstoryline-use ~/.claude/skills/
```

### Other Compatible Agents (Experimental)

These Skills are based on an open Agent Skills format, so in theory they can also be installed into other compatible agents.
For example, you can install them into Codex via the Skills CLI:

```bash
npx skills add FireRedTeam/FireRed-OpenStoryline --skill openstoryline-install --agent codex
npx skills add FireRedTeam/FireRed-OpenStoryline --skill openstoryline-use --agent codex
```

Or use the commands below with the `--global` flag to install these Skills into the user-level directory so they are available across projects:

```bash
npx skills add FireRedTeam/FireRed-OpenStoryline --skill openstoryline-install --global
npx skills add FireRedTeam/FireRed-OpenStoryline --skill openstoryline-use --global
```

## 📦 Install
### 1. Clone repository
```
# If git is not installed, refer to the official website for installation: https://git-scm.com/install/
# Or manually download the code
git clone https://github.com/FireRedTeam/FireRed-OpenStoryline.git
cd FireRed-OpenStoryline
```
### 2. Create a virtual environment

Install Conda according to the official guide (Miniforge is recommended, it is suggested to check the option to automatically configure environment variables during installation): https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html


```
# Recommended python>=3.11
conda create -n storyline python=3.11
conda activate storyline
```
### 3. 📦 Resource Download & Installation
#### 3.1 Automatic Installation (Linux and macOS only)
```
sh build_env.sh
```
#### 3.2 Manual Installation
##### A. MacOS or Linux
  - Step 1: Install wget (if not already installed)

    ```
    # MacOS: If you haven't installed Homebrew yet, please install it first: https://brew.sh/
    brew install wget
    
    # Ubuntu/Debian
    sudo apt-get install wget
    
    # CentOS
    sudo yum install wget
    ```
  - Step 2: Download Resources

    ```bash
    chmod +x download.sh
    ./download.sh
    ```
  
  - Step 3: Install Dependencies

    ```bash
    pip install -r requirements.txt
    ```

##### B. Windows

  - Step 1: Prepare Directory: Create a new directory named `resource` in the project root directory.

  - Step 2: Download and Extract:

    *   [Download Models (models.zip)](https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/models.zip) -> Extract to the `.storyline` directory.

    *   [Download Resources (resource.zip)](https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/resource.zip) -> Extract to the `resource` directory.
  - Step 3:  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Quick Start

Note: Before starting, you need to configure the API-Key in config.toml first. For details, please refer to the documentation [API-Key Configuration](docs/source/en/api-key.md)


### 1. Start the MCP Server

#### MacOS or Linux

```bash
PYTHONPATH=src python -m open_storyline.mcp.server
```

#### Windows
```
$env:PYTHONPATH="src"; python -m open_storyline.mcp.server
```

### 2. Start the conversation interface

- Method 1: Command Line Interface

  ```bash
  python cli.py
  ```

- Method 2: Web Interface

  ```bash
  uvicorn agent_fastapi:app --host 127.0.0.1 --port 8005
  ```

## 🐳 Docker

### Pull the Image
```bash
# Pull image from Docker Hub official repository
# Recommended for users outside China
docker pull openstoryline/openstoryline:v1.0.1

# Pull image from Alibaba Cloud Container Registry
# Recommended for users in China (faster and more stable)
docker pull crpi-6knxem4w8ggpdnsn.cn-shanghai.personal.cr.aliyuncs.com/openstoryline/openstoryline:v1.0.1
```

### Start the Container
```
docker run \
  -v $(pwd)/config.toml:/app/config.toml \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/run.sh:/app/run.sh \
  -p 7860:7860 \
  openstoryline/openstoryline:v1.0.1
```
After starting, access the Web interface at http://0.0.0.0:7860

## 📁 Project Structure
```
FireRed-OpenStoryline/
├── 🎯 src/open_storyline/           Core application
│   ├── mcp/                         🔌 Model Context Protocol
│   ├── nodes/                       🎬 Video processing nodes
│   ├── skills/                      🛠️ Agent skills library
│   ├── storage/                     💾 Agent Memory
│   ├── utils/                       🧰 Helper utilities
│   ├── agent.py                     🤖 Build Agent
│   └── config.py                    ⚙️ Configuration management
├── 📚 docs/                         Documentation
├── 🐳 Dockerfile                    Docker Configuration
├── 💬 prompts/                      LLM prompt templates
├── 🎨 resource/                     Static resources
│   ├── bgms/                        Background music library
│   ├── fonts/                       Font files
│   ├── script_templates/            Video script templates
│   └── unicode_emojis.json          Emoji list
├── 🔧 scripts/                      Utility scripts
├── 🌐 web/                          Web interface
├── 🚀 agent_fastapi.py              FastAPI server
├── 🖥️ cli.py                        Command-line interface
├── ⚙️ config.toml                   Main configuration file
├── 🚀 build_env.sh                  Environment Build Script
├── 📥 download.sh                   Resource downloader
├── 📦 requirements.txt              Runtime dependencies
└── ▶️ run.sh                        Launch script

```

## 📚 Documentation

### 📖 Tutorial Index

- [API Key Configuration](docs/source/en/api-key.md) - How to configure and manage API keys
- [Usage Tutorial](docs/source/en/guide.md) - Common use cases and basic operations
- [FAQ](docs/source/en/faq.md) - Frequently asked questions

## TODO

- [ ] Add the function of **voiceover type video editing**.
- [ ] Add support for **voice cloning**
- [ ] Add more **transition/filter/effects** effects functions.
- [ ] Add **image/video generation and editing** capabilities.
- [ ] **GPU-accelerated** rendering and highlight selection.

## Acknowledgements

This project is built upon the following excellent open-source projects:

### Core Dependencies
- [MoviePy](https://github.com/Zulko/moviepy) - Video editing library
- [FFmpeg](https://ffmpeg.org/) - Multimedia framework
- [LangChain](https://www.langchain.com/) - A framework that provides pre-built Agents

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⭐ Star History

<div align="center"> <p> <img width="800" src="https://api.star-history.com/svg?repos=FireRedTeam/FireRed-OpenStoryline&type=Date" alt="Star-history"> </p> </div>
