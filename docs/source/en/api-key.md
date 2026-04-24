# API Key Configuration Guide

## 1. Quick Start: Provider Presets Are Now Available in the Web UI

The web UI `LLM` / `VLM` panel now supports four provider presets:

- `OpenAI`
- `Gemini`
- `NVIDIA`
- `Ollama`

How to use it:

1. First fill the provider API keys in `config.toml` under `model_providers`
2. In the web UI `LLM model` or `VLM model` dropdown, directly choose `OpenAI / Gemini / NVIDIA / Ollama`
3. The app auto-fills the default model
4. You only need to adjust the `model` field when needed; you no longer enter `API keys` or `Base URLs` in the UI

Notes:

- `API keys` and `Base URLs` are now read from `config.toml` `[model_providers.<provider>]`
- Local `Ollama` mode automatically uses `api_key = "ollama"` as a placeholder
- The safest configurations right now are:
  - `Gemini` for `LLM + VLM`
  - or `NVIDIA` for `LLM` and `Gemini` for `VLM`

## 2. Four Copy-Paste-Ready `config.toml` Templates

### Option A: OpenAI for both LLM + VLM

Replace both placeholder values with your real key:

```toml
[llm]
model = "gpt-4.1-mini"
base_url = "https://api.openai.com/v1"
api_key = "<set-your-key-here>"
timeout = 30.0
temperature = 0.1
max_retries = 2

[vlm]
model = "gpt-4.1-mini"
base_url = "https://api.openai.com/v1"
api_key = "<set-your-key-here>"
timeout = 20.0
temperature = 0.1
max_retries = 2
```

Best for:

- users who already have `OPENAI_API_KEY`
- using one model for both text and image understanding
- avoiding multi-provider mixing

### Option B: Gemini for both LLM + VLM

Replace both placeholder values with your real key:

```toml
[llm]
model = "gemini-3-flash-preview"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key = "<set-your-key-here>"
timeout = 30.0
temperature = 0.1
max_retries = 2

[vlm]
model = "gemini-3-flash-preview"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key = "<set-your-key-here>"
timeout = 20.0
temperature = 0.1
max_retries = 2
```

Best for:

- the fastest setup
- using one key for both text and multimodal tasks
- keeping configuration simple

### Option C: NVIDIA for LLM, Gemini for VLM

This is the more reliable hybrid setup today. Replace the placeholder values:

```toml
[llm]
model = "nvidia/nemotron-mini-4b-instruct"
base_url = "https://integrate.api.nvidia.com/v1"
api_key = "<set-your-key-here>"
timeout = 30.0
temperature = 0.1
max_retries = 2

[vlm]
model = "gemini-3-flash-preview"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key = "<set-your-key-here>"
timeout = 20.0
temperature = 0.1
max_retries = 2
```

Notes:

- `NVIDIA` currently fits the OpenStoryline `LLM` path more naturally
- If you want to test the `NVIDIA` preset for `VLM` in the web UI, you can still switch to it, but you may need to manually adjust the model based on what your account supports

### Option D: Local Ollama with Gemma 4 for LLM + VLM

First pull the model:

```bash
ollama pull gemma4
```

Then use this `config.toml` block. No real cloud API key is needed:

```toml
[llm]
model = "gemma4"
base_url = "http://127.0.0.1:11434/v1"
api_key = "ollama"
timeout = 30.0
temperature = 0.1
max_retries = 2

[vlm]
model = "gemma4"
base_url = "http://127.0.0.1:11434/v1"
api_key = "ollama"
timeout = 20.0
temperature = 0.1
max_retries = 2
```

Best for:

- local development
- reducing cloud costs
- offline debugging of the agent / MCP / node pipeline

## 3. More LLM Notes

## 4. Large Language Model (LLM)

### Using DeepSeek as an Example

**Official Documentation**: https://api-docs.deepseek.com/zh-cn/

Note: For users outside China, we recommend using large language models such as Gemini, Claude, or ChatGPT for the best experience.

### Configuration Steps

1. **Apply for API Key**
   - Visit platform: https://platform.deepseek.com/usage
   - Log in and apply for API Key
   - ⚠️ **Important**: Save the obtained API Key securely

2. **Configuration Parameters**
   - **Model Name**: `deepseek-chat`
   - **Base URL**: `https://api.deepseek.com/v1`
   - **API Key**: Fill in the Key obtained in the previous step

3. **API Configuration**
   - **Web Usage**:
      - In the LLM model dropdown, select **Custom Model**, then fill in the model settings according to your configuration parameters.
      - Or, open `config.toml`, locate `[llm]`, and configure `model`, `base_url`, and `api_key`. The model you entered will then appear in the dropdown on the Web page.
   - **CLI**:
      - If you prefer the CLI entry point, you need to open `config.toml`, locate `[llm]`, and configure `model`, `base_url`, and `api_key`.


## 5. Multimodal Large Language Model (VLM)

### 2.1 Using GLM-4.6V

**API Key Management**: https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys

### Configuration Parameters

- **Model Name**: `glm-4.6v`
- **Base URL**: `https://open.bigmodel.cn/api/paas/v4/`

### 2.2 Using Qwen3-VL

**API Key Management**: Go to Alibaba Cloud Bailian Platform to apply for an API Key https://bailian.console.aliyun.com/cn-beijing/?apiKey=1&tab=globalset#/efm/api_key

  - **Model Name**: `qwen3-vl-8b-instruct`
  - **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`

  - Parameter Configuration: 
    - **Web Usage**:
      - In the VLM model dropdown, select **Custom Model**, then fill in the model settings according to your configuration parameters.
      - Or, open `config.toml`, locate `[vlm]`, and configure `model`, `base_url`, and `api_key`. The model you entered will then appear in the dropdown on the Web page.
    - **CLI**: 
      - If you prefer the CLI entry point, you need to open `config.toml`, locate `[vlm]`, and configure `model`, `base_url`, and `api_key`.

### 2.3 Using Qwen3-Omni

Qwen3-Omni can also be applied for through the Alibaba Cloud Bailian Platform. The specific parameters are as follows, which can be used for automatic labeling music in omni_bgm_label.py
- **Model Name**: `qwen3-omni-flash-2025-12-01`
- **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`

For more details, please refer to the documentation: https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc

Model List: https://help.aliyun.com/zh/model-studio/models

Billing Dashboard: https://billing-cost.console.aliyun.com/home

## 6. Pexels Image and Video Download API Key Configuration

1. Open the Pexels website, register an account, and apply for an API key at https://www.pexels.com/api/
<div align="center">
  <img src="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/docs/resource/pexels_api.png" alt="Pexels API application" width="70%">
  <p><em>Figure 1: Pexels API Application Page</em></p>
</div>

2. Web Usage: Locate the Pexels configuration, select "Use custom key", and enter your API key in the form.
<div align="center">
  <img src="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/docs/resource/use_pexels_api_en.png" alt="Pexels API input" width="70%">
  <p><em>Figure 2: Pexels API Usage</em></p>
</div>

3. Local Deployment: Fill in the API key in the `pexels_api_key` field in the `config.toml` file as the default configuration for the project.

## 7. TTS (Text-to-Speech) Configuration



### Option 1: MiniMax (Recommended)

- **Service URL**: https://platform.minimaxi.com/docs/api-reference/speech-t2a-http
- **API Key Base Url**: https://api.minimax.chat/v1/t2a_v2

- **Configuration Steps**:
   1. Create API Key
   2. Visit: https://platform.minimax.io/user-center/basic-information/interface-key
   3. Obtain and save API Key

### Option 2: Bytedance (Recommended)
1. Step 1: Enable Audio/Video Subtitle Generation Service
   Use the legacy page to find the audio/video subtitle generation service:

   - Visit: https://console.volcengine.com/speech/service/9?AppID=8782592131

2. Step 2: Obtain Authentication Information
   View the account basic information page:
   
   - Visit: https://console.volcengine.com/user/basics/

<div align="center">
  <img src="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/docs/resource/use_bytedance_tts_zh.png" alt="Bytedance TTS API Configuration" width="70%">
  <p><em>Figure 3: Bytedance TTS API Usage</em></p>
</div>

   You need to obtain the following information:
   - **UID**: The ID from the main account information
   - **APP ID**: The APP ID from the service interface authentication information
   - **Access Token**: The Access Token from the service interface authentication information
   
   For local deployment, modify the config.toml file:

```
[generate_voiceover.providers.bytedance]
uid = ""
appid = ""
access_token = ""
```

For detailed documentation, please refer to: https://www.volcengine.com/docs/6561/80909

### Option 3: 302.ai (Alternative solutions)

- **Service URL**: https://302.ai/product/detail/302ai-mmaudio-text-to-speech
- **API Key Base url**：https://api.302.ai

## 8. AI Transition Configuration

**Before you start**: AI transitions trigger additional model calls. Transitions are generated clip by clip between adjacent segments, so the more clips you have and the finer the shot splitting is, the higher the number of calls will usually be. As a result, resource usage is typically **significantly higher** than standard copywriting or voiceover workflows.

**Output quality note**: The current transition description is generated from the first and last frames of adjacent clips by a vision model, while clip ordering is determined by the language model. Final results can therefore vary depending on frame content, prompts, model versions, and service-side behavior. Some randomness is expected, and output may not match expectations every time.

**Recommendation**: Start with a small test run, review the results, and then scale up if the quality and cost are acceptable. Please also check your **account balance** and **provider billing rules** in advance.

### Option 1: MiniMax Hailuo

1. In most cases, the API key you already use for MiniMax LLM or TTS services can also be used for Hailuo video generation. If you already have one, you can reuse it directly. If not, create one from the MiniMax API platform by following the official [Quick Start](https://platform.minimax.io/docs/guides/quickstart).

2. You can use `MiniMax-Hailuo-02`, or check the official [Video Generation documentation](https://platform.minimax.io/docs/api-reference/video-generation-intro) for newer supported model names.

### Option 2: Alibaba Cloud Wan

1. In most cases, the API key you already use for Alibaba Cloud Model Studio LLM services can also be used for Wan video generation. If you already have one, you can reuse it directly. If not, follow the official guide to [get an API key](https://www.alibabacloud.com/help/en/model-studio/get-api-key).

2. We recommend `wan2.2-kf2v-flash`, or you can check the official [first-and-last-frame image-to-video guide](https://www.alibabacloud.com/help/en/model-studio/image-to-video-first-and-last-frames-guide) for more supported model names and usage details.

## Important Notes

- All API Keys must be kept secure to avoid leakage
- Ensure sufficient account balance before use
- Regularly monitor API usage and costs
