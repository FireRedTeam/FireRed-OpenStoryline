# API Key Configuration Guide

## 0. Atlas Cloud — One Key for Both LLM & VLM (OpenAI-compatible, recommended)

FireRed-OpenStoryline talks to its `[llm]` and `[vlm]` backends through a standard
OpenAI-compatible `chat/completions` endpoint. [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=FireRed-OpenStoryline)
exposes exactly that interface, so a **single `base_url` + API key** can serve both the
text LLM (script planning / scheduling) and the multimodal VLM (clip understanding) — no
separate vendor accounts.

1. **Get an API Key**: sign in at [atlascloud.ai](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=FireRed-OpenStoryline)
   and create a key. Keep it secret.
2. **Configuration parameters**
   - **Base URL**: `https://api.atlascloud.ai/v1`
   - **LLM model**: `deepseek-ai/deepseek-v4-pro` (a reasoning model — keep `max_tokens` large enough, e.g. ≥ 512)
   - **VLM model**: `qwen/qwen3-vl-30b-a3b-instruct` (or `qwen/qwen3-vl-8b-instruct` for a lighter option)
   - **API Key**: the key from step 1
3. **Fill it into `config.toml`** (same place as any other OpenAI-compatible provider):

   ```toml
   [llm]
   model = "deepseek-ai/deepseek-v4-pro"
   base_url = "https://api.atlascloud.ai/v1"
   api_key = ""   # your Atlas Cloud key

   [vlm]
   model = "qwen/qwen3-vl-30b-a3b-instruct"
   base_url = "https://api.atlascloud.ai/v1"
   api_key = ""   # the same Atlas Cloud key works here too
   ```

   On the Web UI you can instead pick **Custom Model** in the LLM / VLM dropdown and paste
   the same `model` / `base_url` / `api_key`.

Atlas Cloud is a full-modal, OpenAI-compatible inference platform: beyond the two models
above it also serves GLM, Kimi, MiniMax, Claude, Gemini and more behind the same endpoint,
plus image / video generation APIs you can reuse for the AI-transition step. Browse the
full catalog at [atlascloud.ai/models](https://www.atlascloud.ai/models).

<details>
<summary>All Atlas Cloud chat models (59)</summary>

- Anthropic (Claude): `anthropic/claude-haiku-4.5-20251001`, `anthropic/claude-opus-4.8`, `anthropic/claude-sonnet-4.6`
- OpenAI (GPT): `openai/gpt-5.4`, `openai/gpt-5.5`
- Google (Gemini): `google/gemini-3.1-flash-lite`, `google/gemini-3.1-pro-preview`, `google/gemini-3.5-flash`
- Alibaba (Qwen): `qwen/qwen2.5-7b-instruct`, `Qwen/Qwen3-235B-A22B-Instruct-2507`, `qwen/qwen3-235b-a22b-thinking-2507`, `qwen/qwen3-30b-a3b`, `Qwen/Qwen3-30B-A3B-Instruct-2507`, `qwen/qwen3-30b-a3b-thinking-2507`, `qwen/qwen3-32b`, `qwen/qwen3-8b`, `Qwen/Qwen3-Coder`, `qwen/qwen3-coder-next`, `qwen/qwen3-max-2026-01-23`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-Next-80B-A3B-Thinking`, `Qwen/Qwen3-VL-235B-A22B-Instruct`, `qwen/qwen3-vl-235b-a22b-thinking`, `qwen/qwen3-vl-30b-a3b-instruct`, `qwen/qwen3-vl-30b-a3b-thinking`, `qwen/qwen3-vl-8b-instruct`, `qwen/qwen3.5-122b-a10b`, `qwen/qwen3.5-27b`, `qwen/qwen3.5-35b-a3b`, `qwen/qwen3.5-397b-a17b`, `qwen/qwen3.6-35b-a3b`, `qwen/qwen3.6-plus`
- DeepSeek: `deepseek-ai/deepseek-ocr`, `deepseek-ai/deepseek-r1-0528`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-V3.1`, `deepseek-ai/DeepSeek-V3.1-Terminus`, `deepseek-ai/deepseek-v3.2`, `deepseek-ai/DeepSeek-V3.2-Exp`, `deepseek-ai/deepseek-v4-flash`, `deepseek-ai/deepseek-v4-pro`
- Moonshot (Kimi): `moonshotai/Kimi-K2-Instruct`, `moonshotai/Kimi-K2-Instruct-0905`, `moonshotai/Kimi-K2-Thinking`, `moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`
- Zhipu (GLM): `zai-org/GLM-4.6`, `zai-org/glm-4.7`, `zai-org/glm-5`, `zai-org/glm-5-turbo`, `zai-org/glm-5.1`, `zai-org/glm-5v-turbo`
- MiniMax: `MiniMaxAI/MiniMax-M2`, `minimaxai/minimax-m2.1`, `minimaxai/minimax-m2.5`, `minimaxai/minimax-m2.7`
- xAI (Grok): `xai/grok-4.3`
- Kuaishou (KAT): `kwaipilot/kat-coder-pro-v2`
- Other: `owl`

</details>

## 1. Large Language Model (LLM)

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


## 2. Multimodal Large Language Model (VLM)

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

## 3. Pexels Image and Video Download API Key Configuration

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

## 4. TTS (Text-to-Speech) Configuration



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

## 5. AI Transition Configuration

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