"""
MiniMax Voice Clone + Voiceover Node
=====================================
Replaces generate_voiceover when the user wants to narrate with a cloned voice.

Pipeline position
-----------------
  group_clips ──┐
                ├──► voice_clone_minimax ──► plan_timeline ──► render_video
  generate_script ─┘

This node shares node_kind="tts" with GenerateVoiceoverNode so plan_timeline
(which require_prior_kind=["tts"]) can treat them interchangeably.

What it does
------------
1. Clone voice  – upload clone_audio (and optional prompt_audio) to MiniMax,
                  get back a persistent voice_id.
2. Generate TTS – for every group_script, call MiniMax T2A v2 with the cloned
                  voice_id, save each wav, return the standard voiceover list.

Output format  (identical to GenerateVoiceoverNode)
------------------------------------------------------
{
  "voiceover": [
    {"voiceover_id": "voiceover_0001", "group_id": "group_0001",
     "path": "/…/voiceover_0001_<ts>.wav", "duration": 3500},
    …
  ]
}

File input format (clone_audio / prompt_audio)
-----------------------------------------------
List-of-dicts, same convention as load_media.inputs, so both local-path and
remote base64 transport are handled transparently by BaseNode.load_inputs_from_client:

  Local:  [{"path": "/abs/path/audio.mp3"}]
  Remote: [{"path": "audio.mp3", "base64": "<b64>", "md5": "<md5>"}]

After load_inputs_from_client, item["path"] is already a server-local path.

API reference: https://platform.minimaxi.com/docs/guides/speech-voice-clone
"""

from __future__ import annotations

import asyncio
import binascii
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type

import requests
from pydantic import BaseModel

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_schema import VoiceCloneMinimaxInput
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.logging import get_logger
from open_storyline.utils.register import NODE_REGISTRY

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "https://api.minimaxi.com"
_MILLISECONDS_PER_SECOND = 1000.0


@NODE_REGISTRY.register()
class VoiceCloneMinimaxNode(BaseNode):
    """
    Clone a voice then generate voiceover for all script groups with that voice.

    node_kind="tts" makes this node a drop-in replacement for GenerateVoiceoverNode
    from the perspective of downstream nodes (plan_timeline, render_video).
    """

    meta = NodeMeta(
        name="voice_clone_minimax",
        description=(
            "Clone a voice using MiniMax's Voice Clone API, then generate voiceover "
            "for every script group using the cloned voice. "
            "Use this instead of generate_voiceover when the user wants narration in "
            "their own voice or a specific person's voice. "
            "Requires clone_audio (mp3/m4a/wav, 10s–5min, ≤20MB). "
            "Optionally accepts prompt_audio (<8s) to improve clone quality. "
            "Output format is identical to generate_voiceover and feeds directly into plan_timeline."
        ),
        node_id="voice_clone_minimax",
        node_kind="tts",  # same kind as GenerateVoiceoverNode
        require_prior_kind=["group_clips", "generate_script"],
        default_require_prior_kind=["group_clips", "generate_script"],
        next_available_node=["plan_timeline", "select_bgm"],
        priority=5,
    )

    input_schema: ClassVar[Type[BaseModel]] = VoiceCloneMinimaxInput

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def default_process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Skip mode: return empty voiceover list (same shape as GenerateVoiceoverNode)."""
        node_state.node_summary.info_for_user(
            "[voice_clone_minimax] Skipped — no voiceover generated."
        )
        return {"voiceover": []}

    async def process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Auto mode: clone voice, then generate TTS for every group script."""

        # ---- 1. Resolve credentials ---------------------------------------
        api_key, base_url = self._resolve_credentials(inputs, node_state)

        # ---- 2. Get group scripts from upstream generate_script -----------
        group_scripts = (inputs.get("generate_script") or {}).get("group_scripts") or []
        if not isinstance(group_scripts, list) or not group_scripts:
            node_state.node_summary.info_for_user(
                "[voice_clone_minimax] No group_scripts found — skipping voiceover generation."
            )
            return {"voiceover": []}

        # ---- 3. Extract clone_audio (already decoded by load_inputs_from_client) --
        clone_audio_list: List[Dict[str, Any]] = inputs.get("clone_audio") or []
        if not clone_audio_list:
            raise ValueError(
                "clone_audio is required. "
                'Pass [{"path": "/abs/path/audio.mp3"}] in local mode, or '
                '[{"path": "filename.mp3", "base64": "...", "md5": "..."}] in remote/web mode.'
            )
        clone_audio_file = Path(clone_audio_list[0]["path"])
        if not clone_audio_file.exists():
            raise FileNotFoundError(
                f"Clone audio file not found on server: {clone_audio_file}"
            )

        # ---- 4. Extract optional prompt_audio -----------------------------
        prompt_audio_list: List[Dict[str, Any]] = inputs.get("prompt_audio") or []
        prompt_audio_file: Optional[Path] = None
        if prompt_audio_list:
            p = Path(prompt_audio_list[0]["path"])
            if p.exists():
                prompt_audio_file = p
            else:
                node_state.node_summary.add_warning(
                    f"[voice_clone_minimax] prompt_audio not found, skipping: {p}",
                    artifact_id=node_state.artifact_id,
                )

        # ---- 5. Determine voice_id ----------------------------------------
        custom_voice_id = (inputs.get("voice_id") or "").strip()
        if not custom_voice_id:
            custom_voice_id = f"cloned_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            node_state.node_summary.info_for_user(
                f"[voice_clone_minimax] No voice_id provided — auto-generated: {custom_voice_id}"
            )

        # ---- 6. Prepare output directory ----------------------------------
        output_dir = self._prepare_output_directory(node_state)

        # ---- 7. Clone voice (blocking HTTP, run in thread) ----------------
        cloned_voice_id = await asyncio.to_thread(
            self._clone_voice_sync,
            api_key=api_key,
            base_url=base_url,
            clone_audio_file=clone_audio_file,
            voice_id=custom_voice_id,
            prompt_audio_file=prompt_audio_file,
            prompt_text=(inputs.get("prompt_text") or "").strip(),
            node_state=node_state,
        )

        # ---- 8. Generate TTS for every group script -----------------------
        model = (inputs.get("model") or "speech-02-hd").strip()
        speed = float(inputs.get("speed") or 1.0)
        ts_ms = int(time.time() * 1000)
        voiceover: List[Dict[str, Any]] = []

        for i, group in enumerate(group_scripts, start=1):
            group_id = (group or {}).get("group_id", "")
            raw_text = (group or {}).get("raw_text", "")

            if not group_id:
                raise ValueError(f"Missing group_id in group_scripts[{i}]: {group}")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(
                    f"raw_text is empty for group_id={group_id}, cannot generate speech."
                )

            voiceover_id = f"voiceover_{i:04d}"
            wav_path = output_dir / f"{voiceover_id}_{ts_ms}.wav"

            await asyncio.to_thread(
                self._tts_minimax_sync,
                api_key=api_key,
                base_url=base_url,
                text=raw_text,
                voice_id=cloned_voice_id,
                model=model,
                speed=speed,
                wav_path=wav_path,
            )

            duration = self._audio_duration_ms(wav_path)
            voiceover.append(
                {
                    "voiceover_id": voiceover_id,
                    "group_id": group_id,
                    "path": str(wav_path),
                    "duration": duration,
                }
            )
            node_state.node_summary.info_for_user(
                f"[voice_clone_minimax] Generated {voiceover_id} ({duration}ms)",
                preview_urls=[str(wav_path)],
            )

        node_state.node_summary.info_for_user(
            f"[voice_clone_minimax] Done. voice_id={cloned_voice_id}, "
            f"{len(voiceover)} voiceover segment(s) generated."
        )
        return {"voiceover": voiceover}

    # ------------------------------------------------------------------
    # Step 1: Voice cloning
    # ------------------------------------------------------------------

    def _clone_voice_sync(
        self,
        *,
        api_key: str,
        base_url: str,
        clone_audio_file: Path,
        voice_id: str,
        prompt_audio_file: Optional[Path],
        prompt_text: str,
        node_state: NodeState,
    ) -> str:
        """Upload audio, call /v1/voice_clone, return the confirmed voice_id."""

        headers_auth = {"Authorization": f"Bearer {api_key}"}
        upload_url = base_url.rstrip("/") + "/v1/files/upload"

        # Upload clone audio
        node_state.node_summary.info_for_user(
            f"[voice_clone_minimax] Uploading clone audio: {clone_audio_file.name}"
        )
        file_id = self._upload_file(
            upload_url=upload_url,
            headers=headers_auth,
            file_path=clone_audio_file,
            purpose="voice_clone",
        )
        node_state.node_summary.info_for_user(
            f"[voice_clone_minimax] Clone audio uploaded, file_id={file_id}"
        )

        # (Optional) Upload prompt audio
        prompt_file_id: Optional[str] = None
        if prompt_audio_file is not None:
            node_state.node_summary.info_for_user(
                f"[voice_clone_minimax] Uploading prompt audio: {prompt_audio_file.name}"
            )
            prompt_file_id = self._upload_file(
                upload_url=upload_url,
                headers=headers_auth,
                file_path=prompt_audio_file,
                purpose="prompt_audio",
            )
            node_state.node_summary.info_for_user(
                f"[voice_clone_minimax] Prompt audio uploaded, file_id={prompt_file_id}"
            )

        # Call voice_clone API
        clone_url = base_url.rstrip("/") + "/v1/voice_clone"
        clone_payload: Dict[str, Any] = {
            "file_id": file_id,
            "voice_id": voice_id,
        }
        if prompt_file_id:
            clone_payload["clone_prompt"] = {
                "prompt_audio": prompt_file_id,
                "prompt_text": prompt_text,
            }

        resp = requests.post(
            clone_url,
            headers={**headers_auth, "Content-Type": "application/json"},
            json=clone_payload,
            timeout=120,
        )
        resp.raise_for_status()
        resp_json = resp.json()

        base_resp = (resp_json or {}).get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (0, None):
            raise RuntimeError(
                f"MiniMax voice_clone API error: status_code={status_code}, "
                f"status_msg={base_resp.get('status_msg')}, response={resp_json}"
            )

        node_state.node_summary.info_for_user(
            f"[voice_clone_minimax] Voice cloned successfully. voice_id={voice_id}"
        )
        return voice_id

    # ------------------------------------------------------------------
    # Step 2: TTS with cloned voice (MiniMax T2A v2)
    # ------------------------------------------------------------------

    def _tts_minimax_sync(
        self,
        *,
        api_key: str,
        base_url: str,
        text: str,
        voice_id: str,
        model: str,
        speed: float,
        wav_path: Path,
    ) -> None:
        """Call MiniMax T2A v2 with the cloned voice_id and save wav to disk."""

        api_url = base_url.rstrip("/") + "/v1/t2a_v2"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "text": text,
            "stream": False,
            "output_format": "hex",
            "voice_setting": {
                "voice_id": voice_id,
                "speed": max(0.5, min(2.0, speed)),
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 24000,
                "bitrate": 128000,
                "format": "wav",
            },
        }

        resp = requests.post(api_url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        resp_json = resp.json()

        base_resp = (resp_json or {}).get("base_resp") or {}
        if base_resp.get("status_code") not in (0, None):
            raise RuntimeError(f"MiniMax TTS failed: {resp_json}")

        data = (resp_json or {}).get("data") or {}
        audio_field = data.get("audio")
        if not audio_field:
            raise RuntimeError(f"MiniMax TTS returned no audio data: {resp_json}")

        if isinstance(audio_field, str) and audio_field.startswith("http"):
            audio_resp = requests.get(audio_field, timeout=120)
            audio_resp.raise_for_status()
            wav_path.write_bytes(audio_resp.content)
        else:
            try:
                wav_path.write_bytes(binascii.unhexlify(audio_field))
            except Exception as e:
                raise RuntimeError(
                    f"MiniMax TTS hex decode failed: {e}, "
                    f"audio_field[:64]={str(audio_field)[:64]}"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_credentials(
        self, inputs: Dict[str, Any], node_state: NodeState
    ) -> tuple[str, str]:
        """
        Resolve MiniMax API key. Priority:
          1. inputs["api_key"]
          2. config.toml [generate_voiceover.providers.minimax].api_key
          3. env MINIMAX_API_KEY / TTS_MINIMAX_API_KEY
        """
        import os

        api_key = (inputs.get("api_key") or "").strip()

        if not api_key:
            try:
                providers = (
                    getattr(self.server_cfg.generate_voiceover, "providers", {}) or {}
                )
                api_key = (
                    (providers.get("minimax") or {}).get("api_key") or ""
                ).strip()
            except Exception:
                pass

        if not api_key:
            for env_var in ("MINIMAX_API_KEY", "TTS_MINIMAX_API_KEY"):
                api_key = (os.getenv(env_var) or "").strip()
                if api_key:
                    break

        if not api_key:
            node_state.node_summary.info_for_llm(
                "MiniMax API key is missing. Ask the user to provide it via the sidebar "
                "or config.toml [generate_voiceover.providers.minimax] api_key."
            )
            raise ValueError(
                "MiniMax API key not found. Set it in config.toml under "
                "[generate_voiceover.providers.minimax] api_key, or via the "
                "MINIMAX_API_KEY environment variable."
            )

        raw_url = (inputs.get("base_url") or "").strip()
        if raw_url:
            # Keep only scheme+host; strip any path that may come from TTS endpoint config
            from urllib.parse import urlparse
            _p = urlparse(raw_url)
            base_url = f"{_p.scheme}://{_p.netloc}" if _p.netloc else raw_url
        else:
            base_url = _DEFAULT_BASE_URL
        return api_key, base_url

    def _upload_file(
        self,
        *,
        upload_url: str,
        headers: Dict[str, str],
        file_path: Path,
        purpose: str,
    ) -> Any:
        """Upload a file to MiniMax /v1/files/upload and return file_id."""
        with open(file_path, "rb") as f:
            resp = requests.post(
                upload_url,
                headers=headers,
                data={"purpose": purpose},
                files={"file": (file_path.name, f)},
                timeout=120,
            )
        resp.raise_for_status()
        resp_json = resp.json()

        base_resp = (resp_json or {}).get("base_resp") or {}
        if base_resp.get("status_code") not in (0, None):
            raise RuntimeError(
                f"MiniMax file upload failed: status_code={base_resp.get('status_code')}, "
                f"status_msg={base_resp.get('status_msg')}"
            )

        file_id = (resp_json.get("file") or {}).get("file_id")
        if not file_id:
            raise RuntimeError(
                f"MiniMax file upload returned no file_id. Response: {resp_json}"
            )
        return file_id

    def _audio_duration_ms(self, audio_path: Path) -> int:
        """Return audio duration in milliseconds."""
        try:
            import librosa

            return int(
                round(
                    librosa.get_duration(path=str(audio_path))
                    * _MILLISECONDS_PER_SECOND
                )
            )
        except Exception as e:
            logger.warning(f"Could not read duration of {audio_path}: {e}")
            return 0
