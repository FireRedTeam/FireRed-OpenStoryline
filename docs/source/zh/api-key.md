# API-Key 配置指南

## 一、大语言模型 (LLM)

### 以 DeepSeek 为例

**官方文档**：https://api-docs.deepseek.com/zh-cn/

提示: 对于中国以外用户建议使用 Gemini、Claude、ChatGPT 等主流大语言模型以获得最佳体验。

### 配置步骤

1. **申请 API Key**
   - 访问平台：https://platform.deepseek.com/usage
   - 登录后申请 API Key
   - ⚠️ **重要**：妥善保存获取的 API Key

2. **配置参数**
   - **模型名称**：`deepseek-chat`
   - **Base URL**：`https://api.deepseek.com/v1`
   - **API Key**：填写上一步获取的 Key

3. **API填写**
   - **Web使用**: 在LLM模型表单中选择使用自定义模型，模型按照配置参数进行填写
   - **本地部署**: 在config.toml中 找到`[developer.chat_models_config."deepseek-chat"]` 将配置参数填写上去，使得Web页面可以访问到该默认配置。 找到`[llm]`并配置model、base_url、api_key

## 二、多模态大模型 (VLM)

### 2.1 使用GLM-4.6V

**API Key 管理**：https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys

### 配置参数

- **模型名称**：`glm-4.6v`
- **Base URL**：`https://open.bigmodel.cn/api/paas/v4/`

### 2.2 使用Qwen3-VL

**API Key管理**：进入阿里云百炼平台申请API Key https://bailian.console.aliyun.com/cn-beijing/?apiKey=1&tab=globalset#/efm/api_key

 - **模型名称**：`qwen3-vl-8b-instruct`
 - **Base URL**：`https://dashscope.aliyuncs.com/compatible-mode/v1`

 - **参数填写**：在VLM Model表单中选择"使用自定义模型"进行参数填写。本地部署时，找到`[vlm]`并配置model、base_url、api_key，在config.toml中新增以下字段作为Web的API默认配置：
    ```
   [developer.chat_models_config."qwen3-vl-8b-instruct"]
   base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
   api_key = "YOUR_API_KEY" 
   timeout = 20.0
   temperature = 0.1
   max_retries = 2
   ```


### 2.3 使用Qwen3-Omni

Qwen3-Omni同样可以在阿里云百炼平台进行申请，具体参数如下，可用于omni_bgm_label.py的音频自动标注
- **模型名称**：`qwen3-omni-flash-2025-12-01`
- **Base URL**：`https://dashscope.aliyuncs.com/compatible-mode/v1`

详细文档参考：https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc

阿里云模型列表：https://help.aliyun.com/zh/model-studio/models

计费看板：https://billing-cost.console.aliyun.com/home

## 三、Pexels 图像和视频下载API密钥配置

1. 打开Pexels网站，注册账号，申请API https://www.pexels.com/zh-cn/api/key/ 
<div align="center">
  <img src="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/docs/resource/pexels_api.png" alt="pexels下载图像和视频API申请" width="70%">
  <p><em>图1: Pexels API申请页面</em></p>
</div>

2. 网页使用：找到Pexels配置，选择使用自定义key，将API key填入表单中。
<div align="center">
  <img src="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/docs/resource/use_pexels_api_zh.png" alt="pexels API填写" width="70%">
  <p><em>图2: Pexels API 使用</em></p>
</div>

3. 本地部署的项目：我们将API填写在config.toml中的pexels_api_key字段中。作为项目的默认配置

## 四、TTS (文本转语音) 配置

### 方案一：302.ai （不推荐）

- **服务地址**：https://302.ai/product/detail/302ai-mmaudio-text-to-speech
- **API Key Base url**：https://api.302.ai

### 方案二：MiniMax（推荐使用）

- **订阅页面**：https://platform.minimax.io/subscribe/audio-subscription
- **API Key Base url**：https://api.minimax.chat/v1/t2a_v2

**配置步骤**：
1. 创建 API Key
2. 访问：https://platform.minimax.io/user-center/basic-information/interface-key
3. 获取并保存 API Key

### 方案三：bytedance （推荐使用）
1. 步骤1：开通音视频字幕生成服务
   使用旧版页面，找到音视频字幕生成服务：
   - 访问：https://console.volcengine.com/speech/service/9?AppID=8782592131

2. 步骤2：获取认证信息
   查看账号基本信息页面：
   - 访问：https://console.volcengine.com/user/basics/

<div align="center">
  <img src="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/openstoryline/docs/resource/use_bytedance_tts_zh.png" alt="Bytedance TTS API填写" width="70%">
  <p><em>图3: Bytedance TTS API 使用</em></p>
</div>

   需要获取以下信息：
   - **UID**: 主账号信息中的 ID
   - **APP ID**: 服务接口认证信息中的 APP ID
   - **Access Token**: 服务接口认证信息中的 Access Token
   
   本地部署使用修改config.toml中
   ```
   [generate_voiceover.providers.bytedance]
   uid = ""
   appid = ""
   access_token = ""
   ```

详细文档请参考：https://www.volcengine.com/docs/6561/80909?lang=zh

## 注意事项

- 所有 API Key 均需妥善保管，避免泄露
- 使用前请确认账户余额充足
- 建议定期检查 API 调用量和费用
