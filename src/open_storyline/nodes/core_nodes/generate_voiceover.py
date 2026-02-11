import os
import asyncio
import base64
import time
import uuid
import binascii
import json
import librosa
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union

import requests

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_schema import GenerateVoiceoverInput
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.parse_json import parse_json_dict
from open_storyline.utils.prompts import get_prompt
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class GenerateVoiceoverNode(BaseNode):
    meta = NodeMeta(
        name="generate_voiceover",
        description="Generate voice-over based on the script",
        node_id="generate_voiceover",
        node_kind="tts",
        require_prior_kind=["group_clips", "generate_script"],
        default_require_prior_kind=["group_clips", "generate_script"],
    )

    input_schema = GenerateVoiceoverInput

    # provider -> handler method name
    _PROVIDER_HANDLERS: Dict[str, str] = {
        "bytedance": "_tts_bytedance_sync",
        "minimax": "_tts_minimax_sync",
        "302": "_tts_302_sync",
    }

    _DEFAULT_PROVIDER = "minimax"

    MILLISECONDS_PER_SECOND = 1000.0
    _SAFE_MARGIN = 10

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        node_state.node_summary.info_for_user("Voiceover not generated")
        return {"voiceover": []}

    async def process(self, node_state: NodeState, inputs: Dict[str, Any], **params) -> Any:
        # 1) Get script
        group_scripts = (inputs.get("generate_script") or {}).get("group_scripts") or []
        if not isinstance(group_scripts, list) or not group_scripts:
            node_state.node_summary.info_for_user("No script found for voiceover generation (group_scripts is empty)")
            return {"voiceover": []}

        # 2) Provider selection
        provider_name = (inputs.get("provider") or "").strip()
        if not provider_name:
            node_state.node_summary.info_for_user("未找到可生成配音的tts提供商，使用默认")

        handler = self._get_provider_handler(provider_name)
        node_state.node_summary.info_for_user(f"TTS 服务：{provider_name}")

        # 3) Prepare output directory
        artifact_id = node_state.artifact_id
        session_id = node_state.session_id
        if not artifact_id or not session_id:
            raise ValueError("缺失 artifact_id / session_id，无法生成配音输出目录")

        output_dir = self.server_cache_dir / str(session_id) / str(artifact_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4) Deduce which key fields this provider needs from config, and get values from inputs
        #    If user/config keys are incomplete, fallback to 302 and use 302 key from environment variables
        try:
            provider_cfg = self._get_provider_cfg(provider_name)
            secrets = self._resolve_provider_secrets(provider_name, provider_cfg, inputs, node_state)
        except ValueError as e:
            if provider_name == self._DEFAULT_PROVIDER:
                raise
            node_state.node_summary.info_for_user(
                f"Key/config for provider={provider_name} is incomplete, automatically falling back to {self._DEFAULT_PROVIDER} (using environment variable key): {e}"
            )
            provider_name = self._DEFAULT_PROVIDER
            handler = self._get_provider_handler(provider_name)
            provider_cfg = self._get_provider_cfg(provider_name)
            secrets = self._resolve_provider_secrets(provider_name, provider_cfg, inputs, node_state)
            node_state.node_summary.info_for_user(f"TTS service fallback to: {provider_name}")

        # 5) Generate parameter dict from provider parameter schema + user_request via LLM
        provider_param_schema = self._load_provider_param_schema(provider_name)
        user_request = inputs.get("user_request", "")
        tts_params = await self._infer_tts_params_with_llm(
            node_state=node_state,
            provider_name=provider_name,
            user_request=user_request,
            provider_param_schema=provider_param_schema,
        )

        if tts_params:
            node_state.node_summary.info_for_user(f"TTS parameters (LLM parsed): {json.dumps(tts_params, ensure_ascii=False)}")
        else:
            node_state.node_summary.info_for_user("TTS parameters: No valid parameters parsed from user_request, using default/server default values")

        # 6) Generate segment by segment
        ts_ms = int(time.time() * 1000)
        voiceover: list[dict[str, Any]] = []

        for i, group in enumerate(group_scripts, start=1):
            group_id = (group or {}).get("group_id", "")
            raw_text = (group or {}).get("raw_text", "")

            if not group_id:
                raise ValueError(f"Missing group_id: {group}")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(f"raw_text is empty for group_id={group_id}, cannot generate speech.")

            voiceover_id = f"voiceover_{i:04d}"
            wav_path = output_dir / f"{voiceover_id}_{ts_ms}.wav"

            await asyncio.to_thread(
                handler,
                text=raw_text,
                wav_path=wav_path,
                secrets=secrets,
                tts_params=tts_params,
                provider_cfg=provider_cfg,
            )

            duration = self._wav_duration_ms(wav_path)
            voiceover.append(
                {
                    "voiceover_id": voiceover_id,
                    "group_id": group_id,
                    "path": str(wav_path),
                    "duration": duration,
                }
            )

            node_state.node_summary.info_for_user(
                f"Successfully generated {voiceover_id}",
                preview_urls=[str(wav_path)],
            )


        node_state.node_summary.info_for_user(f"Generated {len(voiceover)} voiceover segments in total")
        return {"voiceover": voiceover}

    # ---------------------------------------------------------------------
    # Provider dispatch / config helpers
    # ---------------------------------------------------------------------

    def _get_provider_handler(self, provider_name: str) -> Callable[..., None]:
        if provider_name is None or provider_name == "":
            provider_name = self._DEFAULT_PROVIDER
        method_name = self._PROVIDER_HANDLERS.get(provider_name)
        if not method_name:
            raise ValueError(f"Unsupported TTS provider: {provider_name}, currently supported: {list(self._PROVIDER_HANDLERS.keys())}")
        handler = getattr(self, method_name, None)
        if not callable(handler):
            raise ValueError(f"Handler for provider={provider_name} not implemented: {method_name}")
        return handler

    def _get_provider_cfg(self, provider_name: str) -> Dict[str, Any]:
        providers = getattr(self.server_cfg.generate_voiceover, "providers", None) or {}
        cfg = providers.get(provider_name)
        if not isinstance(cfg, dict):
            if provider_name == self._DEFAULT_PROVIDER:
                return {"api_key": "", "base_url": ""}
            raise ValueError(f"provider={provider_name} not configured in server_cfg.generate_voiceover.providers")
         
        return cfg

    def _resolve_provider_secrets(self, provider_name: str, provider_cfg: Dict[str, Any], inputs: Dict[str, Any], node_state: NodeState) -> Dict[str, Any]:
        """
        - Each field uses inputs[field] first, otherwise falls back to cfg[field]
        - base_url can be omitted: default value will be provided based on provider
        """
        secrets: Dict[str, Any] = {}
        required_keys = list(provider_cfg.keys())
        provider_keys = inputs.get("provider_keys") or {}
        if not isinstance(provider_keys, dict):
            provider_keys = {}

        for key in required_keys:
            value = inputs.get(key)
            if value in (None, ""):
                value = provider_keys.get(key)

            if value in (None, ""):
                value = provider_cfg.get(key)

            if (value in (None, "")) and key == "base_url":
                value = self._default_base_url(provider_name)
            
            if (value in (None, "")) and provider_name == self._DEFAULT_PROVIDER:
                env_v = self._resolve_minimax_env_secret(key)
                if env_v not in (None, ""):
                    value = env_v

            if value in (None, ""):
                node_state.node_summary.info_for_llm("The user has not entered the voice-over service API key, please remind the user to enter the TTS API key in the sidebar of the webpage.")
                raise ValueError(
                    f"provider={provider_name} missing required field: {key}. "
                    f"Please configure in sidebar or config.toml."
                )

            secrets[key] = value

        return secrets

    def _default_base_url(self, provider_name: str) -> str:
        if provider_name == "bytedance":
            return "https://openspeech.bytedance.com"
        if provider_name == "minimax":
            return "https://api.minimax.chat"
        if provider_name == "302":
            return "https://api.302.ai"
        return ""

    # ---------------------------------------------------------------------
    # LLM param inference
    # ---------------------------------------------------------------------
    def _load_provider_param_schema(self, provider_name: str) -> Dict[str, Any]:

        path = self.server_cfg.generate_voiceover.tts_provider_params_path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        providers = (data or {}).get("providers") or {}
        schema = providers.get(provider_name) or {}
        return schema if isinstance(schema, dict) else {}

    async def _infer_tts_params_with_llm(
        self,
        node_state: NodeState,
        provider_name: str,
        user_request: Any,
        provider_param_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Pass user_request + provider parameter definition to LLM, let it return JSON dict.
        """
        if not provider_param_schema:
            return {}

        system_prompt = get_prompt("generate_voiceover.system", lang=node_state.lang)

        schema_text = json.dumps(provider_param_schema, ensure_ascii=False, indent=2)

        user_prompt = get_prompt("generate_voiceover.user", lang=node_state.lang, provider_name=provider_name, user_request=str(user_request), schema_text=schema_text)
        raw = await node_state.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            top_p=0.9,
            max_tokens=4096,
            model_preferences=None
        )
        if not raw:
            return {}

        parsed = parse_json_dict(raw)
        if not isinstance(parsed, dict):
            return {}

        return self._sanitize_params_by_schema(parsed, provider_param_schema)

    # ---------------------------------------------------------------------
    # validation helpers
    # ---------------------------------------------------------------------

    def _resolve_302_env_secret(self, key: str) -> Optional[str]:
        """
        Read 302 key/config from environment variables
        """
        key = str(key).strip()
        if not key:
            return None

        key_upper = key.upper()
        prefixe = ("TTS_302_")

        return os.getenv(f"{prefixe}{key_upper}")
    
    def _resolve_minimax_env_secret(self, key: str) -> Optional[str]:
        """
        从环境变量读取 minimax 的密钥/配置
        """
        key = str(key).strip()
        if not key:
            return None

        key_upper = key.upper()
        prefixe = ("TTS_MINIMAX_")

        return os.getenv(f"{prefixe}{key_upper}")
  
    def _sanitize_params_by_schema(self, params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        - Only keep fields that exist in schema
        - Type coercion
        - value validation (string enum / numeric range / discrete numeric enum)
        """
        out: Dict[str, Any] = {}

        for key, val in params.items():
            if key not in schema:
                continue

            rule = schema.get(key) or {}
            if not isinstance(rule, dict):
                continue

            typ = (rule.get("type") or "").lower().strip()
            normalized = self._normalize_value(val, typ)
            if normalized is None:
                continue

            # 1) Continuous: range: [min, max]
            if "range" in rule:
                vaule_range = rule.get("range")
                if (
                    typ in ("int", "float")
                    and isinstance(vaule_range, list)
                    and len(vaule_range) == 2
                    and all(isinstance(x, (int, float)) for x in vaule_range)
                ):
                    range_min, range_max = float(vaule_range[0]), float(vaule_range[1])
                    value = float(normalized)

                    if value < range_min:
                        value = range_min
                    elif value > range_max:
                        value = range_max

                    normalized = int(value) if typ == "int" else float(round(value, 1))

                else:
                    continue

            # 2) Discrete: enum: [...]
            elif "enum" in rule:
                enum = rule.get("enum")
                if isinstance(enum, list) and enum:
                    if normalized not in enum:
                        normalized = enum[0]
                else:
                    continue

            # 3) No range/enum: keep normalized as-is (type-coerced only)
            out[key] = normalized

        return out
    
    def _normalize_value(self, val: Any, typ: str) -> Any:
        if typ in ("str", "string"):
            return str(val)

        if typ in ("int", "integer"):
            return int(val)

        if typ in ("float"):
            return float(val)

        if typ in ("bool", "boolean"):
            return bool(val)

        return val
    
    def _wav_duration_ms(self, wav_path: Union[str, Path]) -> int:
        p = str(wav_path)

        duration_s = librosa.get_duration(path=p)
        return int(round(duration_s * self.MILLISECONDS_PER_SECOND))

    # ---------------------------------------------------------------------
    # Provider implementations (each provider has its own dedicated method)
    # ---------------------------------------------------------------------

    def _preview_b64(self, b64: str, keep: int = 80) -> str:
        if not isinstance(b64, str):
            return f"<non-str data type={type(b64).__name__}>"
        if len(b64) <= keep * 2:
            return b64
        return f"{b64[:keep]}...<len={len(b64)}>...{b64[-keep:]}"

    def _tts_bytedance_sync(
        self,
        *,
        text: str,
        wav_path: Path,
        secrets: Dict[str, Any],
        tts_params: Dict[str, Any],
        provider_cfg: Dict[str, Any],
    ) -> None:

        base_url = secrets.get("base_url") or "https://openspeech.bytedance.com"
        api_url = base_url.rstrip("/") + "/api/v1/tts" if not base_url.endswith("/api/v1/tts") else base_url

        access_token = secrets.get("access_token")
        appid = secrets.get("appid")
        uid = secrets.get("uid")
        cluster = secrets.get("cluster") or "volcano_tts"

        headers = {"Authorization": f"Bearer; {access_token}"}

        audio_cfg = {
            "voice_type": tts_params.get("voice_type", "BV700_streaming"),
            "encoding": tts_params.get("encoding", "wav"),
            "rate": int(tts_params.get("rate", 24000)) if "rate" in tts_params else 24000,
            "speed_ratio": float(tts_params.get("speed_ratio", 1.0)),
            "volume_ratio": float(tts_params.get("volume_ratio", 1.0)),
            "pitch_ratio": float(tts_params.get("pitch_ratio", 1.0)),
        }
        # 可选字段
        for k in ("emotion", "language"):
            if k in tts_params:
                audio_cfg[k] = tts_params[k]

        request_cfg = {
            "reqid": str(uuid.uuid4()),
            "text": text,
            "text_type": tts_params.get("text_type", "plain"),
            "operation": "query",
        }

        body = {
            "app": {"appid": appid, "token": access_token, "cluster": cluster},
            "user": {"uid": uid},
            "audio": audio_cfg,
            "request": request_cfg,
        }

        resp = requests.post(api_url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()

        resp_json = resp.json()
        if isinstance(resp_json, dict):
            code = resp_json.get("code")
            message = resp_json.get("message")
            resp_preview = dict(resp_json)
            b64 = resp_json.get("data")
            if isinstance(b64, str) and len(b64) > 200:
                resp_preview["data"] = self._preview_b64(b64)
            if code not in (3000, 0, None):
                raise RuntimeError(f"bytedance tts failed: code={code}, message={message}, resp={resp_preview}")
            if message not in (None, "Success") and code is None:
                raise RuntimeError(f"bytedance tts failed: message={message}, resp={resp_json}")

            b64 = resp_json.get("data")
            if not b64:
                raise RuntimeError(f"bytedance tts failed: no data in resp={resp_json}")

            audio_bytes = base64.b64decode(b64)
            wav_path.write_bytes(audio_bytes)
            return

        raise RuntimeError(f"bytedance tts failed: invalid resp: {resp.text}")

    def _tts_minimax_sync(
        self,
        *,
        text: str,
        wav_path: Path,
        secrets: Dict[str, Any],
        tts_params: Dict[str, Any],
        provider_cfg: Dict[str, Any],
    ) -> None:
        
        base_url = secrets.get("base_url") or "https://api.minimax.chat"
        api_url = base_url.rstrip("/") + "/v1/t2a_v2" if not base_url.endswith("/v1/t2a_v2") else base_url

        api_key = secrets.get("api_key") or secrets.get("token") or secrets.get("access_token")
        if not api_key:
            for k, v in secrets.items():
                if k != "base_url" and isinstance(v, str) and v.strip():
                    api_key = v.strip()
                    break
        if not api_key:
            raise ValueError("minimax missing api_key/token/access_token")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": tts_params.get("model", "speech-02-hd"),
            "text": text,
            "stream": False,
            "language_boost": tts_params.get("language_boost", "auto"),
            "output_format": tts_params.get("output_format", "hex"),
            "voice_setting": {
                "voice_id": tts_params.get("voice_id", "English_expressive_narrator"),
                "speed": float(tts_params.get("speed", 1.0)),
                "vol": float(tts_params.get("vol", 1.0)),
                "pitch": int(tts_params.get("pitch", 0)),
            },
            "audio_setting": {
                "sample_rate": int(tts_params.get("sample_rate", 24000)),
                "bitrate": int(tts_params.get("bitrate", 128000)),
                "format": tts_params.get("format", "wav"),
            },
        }

        resp = requests.post(api_url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()

        resp_json = resp.json()
        base_resp = (resp_json or {}).get("base_resp") or {}
        if base_resp.get("status_code") not in (0, None):
            raise RuntimeError(f"minimax tts failed: {resp_json}")

        data = (resp_json or {}).get("data") or {}
        audio_field = data.get("audio")
        if not audio_field:
            raise RuntimeError(f"minimax tts failed: no data.audio: {resp_json}")

        # output_format = hex or url
        if isinstance(audio_field, str) and audio_field.startswith("http"):
            audio_bytes = requests.get(audio_field, timeout=120).content
            wav_path.write_bytes(audio_bytes)
            return

        try:
            audio_bytes = binascii.unhexlify(audio_field)
        except Exception as e:
            raise RuntimeError(f"minimax hex decode failed: {e}, audio_field[:64]={str(audio_field)[:64]}")

        wav_path.write_bytes(audio_bytes)

    def _tts_302_sync(
        self,
        *,
        text: str,
        wav_path: Path,
        secrets: Dict[str, Any],
        tts_params: Dict[str, Any],
        provider_cfg: Dict[str, Any],
    ) -> None:
        base_url = (secrets.get("base_url") or "https://api.302.ai").rstrip("/")
        api_url = base_url + "/302/audio/speech"

        api_key = secrets.get("api_key") or secrets.get("token") or secrets.get("access_token")
        if not api_key:
            for k, v in secrets.items():
                if k != "base_url" and isinstance(v, str) and v.strip():
                    api_key = v.strip()
                    break
        if not api_key:
            raise ValueError("302 missing api_key/token/access_token")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "audio/wav",
            "Content-Type": "application/json",
        }

        body = {
            "model": tts_params.get("model", "speech-02-hd"),
            "input": text,
            "voice": tts_params.get("voice", "alloy"),
            "emotion": tts_params.get("emotion", "neutral"),
            "response_format": tts_params.get("response_format", "wav"),
        }

        resp = requests.post(api_url, headers=headers, json=body, timeout=120)
        if not resp.ok:
            raise RuntimeError(f"302 tts http {resp.status_code}: {resp.text}")
        wav_path.write_bytes(resp.content)