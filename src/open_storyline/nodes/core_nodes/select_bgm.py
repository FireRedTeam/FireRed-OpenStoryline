import asyncio
import time
from typing import Any, Dict
from pathlib import Path

import numpy as np
import random
import librosa
import requests

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import SelectBGMInput
from open_storyline.utils.element_filter import ElementFilter
from open_storyline.utils.localai_auth import resolve_localai_shared_token
from open_storyline.utils.recall import StorylineRecall
from src.open_storyline.utils.prompts import get_prompt
from open_storyline.utils.parse_json import parse_json_dict
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class SelectBGMNode(BaseNode):
    meta = NodeMeta(
        name="select_bgm",
        description="Select appropriate BGM based on user requirements",
        node_id="select_bgm",
        node_kind="music_rec",
        require_prior_kind=[],
        default_require_prior_kind=[],
        next_available_node=["plan_timeline"],
    )

    input_schema = SelectBGMInput

    _FILTER_CANONICAL_VALUES = {
        "mood": {
            "dynamic": "Dynamic",
            "chill": "Chill",
            "happy": "Happy",
            "sorrow": "Sorrow",
            "romantic": "Romantic",
            "calm": "Calm",
            "excited": "Excited",
            "healing": "Healing",
            "inspirational": "Inspirational",
        },
        "scene": {
            "vlog": "Vlog",
            "travel": "Travel",
            "relaxing": "Relaxing",
            "emotion": "Emotion",
            "transition": "Transition",
            "outdoor": "Outdoor",
            "cafe": "Cafe",
            "evening": "Evening",
            "scenery": "Scenery",
            "food": "Food",
            "date": "Date",
            "club": "Club",
        },
        "genre": {
            "pop": "Pop",
            "bgm": "BGM",
            "electronic": "Electronic",
            "r&b/soul": "R&B/Soul",
            "hip hop/rap": "Hip Hop/Rap",
            "rock": "Rock",
            "jazz": "Jazz",
            "folk": "Folk",
            "classical": "Classical",
            "chinese style": "Chinese Style",
        },
        "lang": {
            "bgm": "bgm",
            "en": "en",
            "zh": "zh",
            "ko": "ko",
            "ja": "ja",
        },
    }
    _DEFAULT_PROVIDER = "localai"
    _NODE_MODE_VALUES = {"auto", "skip", "default"}

    def __init__(self, server_cfg):
        super().__init__(server_cfg)
        self.element_filter = ElementFilter(json_path=f"{self.server_cfg.project.bgm_dir}/meta.json")
        self.vectorstore = StorylineRecall.build_vectorstore(self.element_filter.library)

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        node_state.node_summary.info_for_user("Failed to choose music")
        return {"bgm": {}}


    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        cfg = self.server_cfg
        provider_name = str(inputs.get("provider") or "").strip().lower()
        if not provider_name:
            provider_name = self._DEFAULT_PROVIDER
        user_request = inputs.get("user_request", "")
        filter_include = self._normalize_filter_map(inputs.get("filter_include", {}))
        filter_exclude = self._normalize_filter_map(inputs.get("filter_exclude", {}))
        if provider_name == "localai":
            session_id = node_state.session_id
            artifact_id = node_state.artifact_id
            if not session_id or not artifact_id:
                raise ValueError("缺失 session_id / artifact_id，无法生成背景音乐输出目录")

            output_dir = self.server_cache_dir / str(session_id) / str(artifact_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            provider_cfg = self._resolve_provider_cfg(provider_name, inputs)
            prompt = self._build_localai_prompt(user_request, filter_include)
            node_state.node_summary.info_for_user("BGM 服务：LocalAI")
            bgm_info = await asyncio.to_thread(
                self._generate_localai_bgm_sync,
                prompt=prompt,
                output_dir=output_dir,
                provider_cfg=provider_cfg,
            )
        else:
            bgm_info = await self.recommend(node_state, user_request, filter_include, filter_exclude)
        if not bgm_info:
            return {"bgm": {}}

        result = self.analyze_music_metrics(bgm_info=bgm_info, sr=cfg.select_bgm.sample_rate, hop_length=cfg.select_bgm.hop_length, frame_length=cfg.select_bgm.frame_length)
        if result.get("path"):
            node_state.node_summary.info_for_user("Successfully choose music", preview_urls=[result.get("path")])
        else:
            node_state.node_summary.info_for_user("Failed to choose music")
        return {"bgm": result}

    def _resolve_provider_cfg(self, provider_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        providers = getattr(self.server_cfg.select_bgm, "providers", None) or {}
        cfg = dict(providers.get(provider_name) or {})
        for key in (
            "mode",
            "base_url",
            "api_key",
            "model_id",
            "duration_seconds",
            "instrumental",
            "prompt_influence",
            "do_sample",
            "think",
            "caption",
            "lyrics",
            "bpm",
            "keyscale",
            "language",
            "vocal_language",
            "timesignature",
        ):
            value = inputs.get(key)
            if key == "mode" and str(value or "").strip().lower() in self._NODE_MODE_VALUES:
                value = None
            if value not in (None, ""):
                cfg[key] = value

        mode = str(cfg.get("mode") or "").strip().lower()
        if not str(cfg.get("base_url") or "").strip():
            cfg["base_url"] = "http://127.0.0.1:18080" if mode == "gateway" else "http://127.0.0.1:8080"
        if "instrumental" not in cfg:
            cfg["instrumental"] = True

        if mode != "gateway" and not str(cfg.get("model_id") or "").strip():
            raise ValueError("LocalAI BGM provider missing model_id. Please configure select_bgm.providers.localai.model_id.")

        return cfg

    @staticmethod
    def _build_localai_prompt(user_request: str, filter_include: Dict[str, Any]) -> str:
        parts = []
        req = str(user_request or "").strip()
        if req:
            parts.append(req)

        if isinstance(filter_include, dict):
            for key in ("mood", "scene", "genre", "lang"):
                values = filter_include.get(key)
                if isinstance(values, list) and values:
                    joined = ", ".join(str(v).strip() for v in values if str(v).strip())
                    if joined:
                        parts.append(f"{key}: {joined}")

        if not parts:
            parts.append("instrumental background music for a short video")

        prompt = "; ".join(parts).strip()
        if "instrumental" not in prompt.lower():
            prompt = f"instrumental background music, {prompt}"
        return prompt

    @classmethod
    def _normalize_filter_map(cls, raw_filters: Any) -> Dict[str, Any]:
        if not raw_filters:
            return {}

        normalized: Dict[str, list[Any]] = {}
        if isinstance(raw_filters, dict):
            for key, values in raw_filters.items():
                name = str(key or "").strip().lower()
                if not name:
                    continue
                bucket = normalized.setdefault(name, [])
                cls._extend_filter_bucket(bucket, values, filter_name=name)
        elif isinstance(raw_filters, list):
            for value in raw_filters:
                cls._classify_filter_value(normalized, value)
        else:
            return {}

        return {k: v for k, v in normalized.items() if isinstance(v, list) and v}

    @classmethod
    def _extend_filter_bucket(cls, bucket: list[Any], values: Any, *, filter_name: str) -> None:
        if isinstance(values, list):
            items = values
        else:
            items = [values]

        for item in items:
            canonical = cls._canonicalize_filter_value(filter_name, item)
            if canonical in (None, ""):
                continue
            if canonical not in bucket:
                bucket.append(canonical)

    @classmethod
    def _classify_filter_value(cls, normalized: Dict[str, list[Any]], value: Any) -> None:
        if isinstance(value, int):
            bucket = normalized.setdefault("id", [])
            if value not in bucket:
                bucket.append(value)
            return

        text = str(value or "").strip()
        if not text:
            return

        for filter_name in ("mood", "scene", "genre", "lang"):
            canonical = cls._canonicalize_filter_value(filter_name, text)
            if canonical not in (None, ""):
                bucket = normalized.setdefault(filter_name, [])
                if canonical not in bucket:
                    bucket.append(canonical)
                return

    @classmethod
    def _canonicalize_filter_value(cls, filter_name: str, value: Any) -> Any:
        if filter_name == "id":
            if isinstance(value, int):
                return value
            text = str(value or "").strip()
            if text.isdigit():
                return int(text)
            return None

        text = str(value or "").strip()
        if not text:
            return None

        canonical_map = cls._FILTER_CANONICAL_VALUES.get(filter_name, {})
        return canonical_map.get(text.lower())

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _suffix_from_content_type(content_type: str) -> str:
        ct = str(content_type or "").split(";", 1)[0].strip().lower()
        return {
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/flac": ".flac",
            "audio/ogg": ".ogg",
        }.get(ct, ".wav")

    def _generate_localai_bgm_sync(
        self,
        *,
        prompt: str,
        output_dir: Path,
        provider_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_url = str(provider_cfg.get("base_url") or "http://127.0.0.1:8080").rstrip("/")
        mode = str(provider_cfg.get("mode") or "").strip().lower()

        if mode == "gateway":
            return self._generate_localai_bgm_gateway_sync(
                prompt=prompt,
                output_dir=output_dir,
                provider_cfg=provider_cfg,
            )

        api_url = base_url + "/v1/sound-generation"

        headers = {
            "Content-Type": "application/json",
        }
        api_key = resolve_localai_shared_token(provider_cfg.get("api_key"))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body: Dict[str, Any] = {
            "model_id": str(provider_cfg.get("model_id") or "").strip(),
            "text": prompt,
            "instrumental": self._coerce_bool(provider_cfg.get("instrumental"), True),
        }

        for key in ("caption", "lyrics", "keyscale", "language", "vocal_language", "timesignature"):
            value = provider_cfg.get(key)
            if value not in (None, ""):
                body[key] = str(value).strip()

        for key, parser in (
            ("duration_seconds", self._coerce_float),
            ("prompt_influence", self._coerce_float),
            ("bpm", self._coerce_int),
        ):
            parsed = parser(provider_cfg.get(key))
            if parsed is not None:
                body[key] = parsed

        for key in ("do_sample", "think"):
            if provider_cfg.get(key) not in (None, ""):
                body[key] = self._coerce_bool(provider_cfg.get(key), False)

        resp = requests.post(api_url, headers=headers, json=body, timeout=300)
        if not resp.ok:
            raise RuntimeError(f"LocalAI sound generation http {resp.status_code}: {resp.text}")

        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = self._suffix_from_content_type(resp.headers.get("Content-Type"))
        generated_id = f"bgm_localai_{int(time.time() * 1000)}"
        output_path = output_dir / f"{generated_id}{suffix}"
        output_path.write_bytes(resp.content)

        return {
            "id": generated_id,
            "bgm_id": generated_id,
            "path": str(output_path),
        }

    def _generate_localai_bgm_gateway_sync(
        self,
        *,
        prompt: str,
        output_dir: Path,
        provider_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_url = str(provider_cfg.get("base_url") or "http://127.0.0.1:18080").rstrip("/")
        api_key = resolve_localai_shared_token(provider_cfg.get("api_key"))
        if not api_key:
            raise ValueError("local-ai-platform music missing api_key / shared token")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "prompt": prompt,
            "style": {
                "instrumental": self._coerce_bool(provider_cfg.get("instrumental"), True),
            },
        }
        duration_seconds = self._coerce_int(provider_cfg.get("duration_seconds"))
        if duration_seconds is not None:
            body["style"]["duration_seconds"] = duration_seconds
        bpm = self._coerce_int(provider_cfg.get("bpm"))
        if bpm is not None:
            body["style"]["bpm"] = bpm
        lyrics = provider_cfg.get("lyrics")
        if lyrics not in (None, ""):
            body["lyrics"] = str(lyrics).strip()

        resp = requests.post(f"{base_url}/api/v1/jobs/music", headers=headers, json=body, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"local-ai-platform music job create http {resp.status_code}: {resp.text}")
        data = resp.json().get("data") or {}
        job_id = str(data.get("job_id") or data.get("id") or "").strip()
        if not job_id:
            raise RuntimeError(f"local-ai-platform music job create returned no job_id: {resp.text}")

        job = self._poll_localai_job(base_url, api_key, job_id, timeout_seconds=300.0)
        result_asset_ids = job.get("result_asset_ids") or []
        if not result_asset_ids:
            raise RuntimeError(f"local-ai-platform music job {job_id} succeeded but returned no result assets")

        asset_id = str(result_asset_ids[0]).strip()
        asset_resp = requests.get(
            f"{base_url}/api/v1/assets/{asset_id}/content",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        if not asset_resp.ok:
            raise RuntimeError(f"local-ai-platform music asset download http {asset_resp.status_code}: {asset_resp.text}")

        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = self._suffix_from_content_type(asset_resp.headers.get("Content-Type"))
        generated_id = f"bgm_localai_{int(time.time() * 1000)}"
        output_path = output_dir / f"{generated_id}{suffix}"
        output_path.write_bytes(asset_resp.content)

        return {
            "id": generated_id,
            "bgm_id": generated_id,
            "path": str(output_path),
        }

    @staticmethod
    def _poll_localai_job(base_url: str, api_key: str, job_id: str, *, timeout_seconds: float) -> Dict[str, Any]:
        deadline = time.time() + max(timeout_seconds, 1.0)
        headers = {"Authorization": f"Bearer {api_key}"}
        job_url = f"{base_url.rstrip('/')}/api/v1/jobs/{job_id}"

        while time.time() < deadline:
            resp = requests.get(job_url, headers=headers, timeout=60)
            if not resp.ok:
                raise RuntimeError(f"local-ai-platform job poll http {resp.status_code}: {resp.text}")
            data = resp.json().get("data") or {}
            status = str(data.get("status") or "").strip().lower()
            if status in {"succeeded", "completed", "success"}:
                return data
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"local-ai-platform job {job_id} failed: {data.get('error') or data}")
            time.sleep(1.0)

        raise TimeoutError(f"local-ai-platform job {job_id} timed out after {timeout_seconds:.0f}s")


    async def recommend(
            self, 
            node_state: NodeState,
            user_request: str, 
            filter_include: Dict={}, 
            filter_exclude: Dict={}
        ):

        # Step1: Check resources
        bgm_dir: Path = self.server_cfg.project.bgm_dir.expanduser().resolve()
        if not bgm_dir.exists():
            raise FileNotFoundError(f"bgm_dir not found: {bgm_dir}")
        if not bgm_dir.is_dir():
            raise NotADirectoryError(f"bgm_dir is not a directory: {bgm_dir}")
        
        # Step2: Full Recall
        candidates = StorylineRecall.query_top_n(self.vectorstore, query=user_request)

        # Step3: Filter tags
        candidates = self.element_filter.filter(candidates, filter_include, filter_exclude)
        if not candidates:
            raise FileNotFoundError(f"No audio files found in: {bgm_dir}")
        
        # Step4: LLM Sampling
        llm = node_state.llm
        system_prompt = get_prompt("select_bgm.system", lang=node_state.lang)
        user_prompt = get_prompt("select_bgm.user", lang=node_state.lang, candidates=candidates, user_request=user_request)
        raw = await llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048,
            model_preferences=None,
        )
        try:
            selected_json = parse_json_dict(raw)
        except:
            selected_json = (raw or "").strip() if raw else "Error: Unable to parse the model output"
            node_state.node_summary.add_error(selected_json)
        
        if not isinstance(selected_json, Dict) or 'path' not in selected_json:
            # Demotion select the first one of candidates
            selected_json = candidates[0]
        
        return selected_json
    

    def analyze_music_metrics(
        self,
        bgm_info: Dict,
        sr: int = 22050,
        hop_length = 2048,
        frame_length = 2048,

    ) -> dict[str, Any]:
        path = Path(bgm_info.get("path"))
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        y, sample_rate = self._load_audio_mono(path, sr=sr)
        duration = int(librosa.get_duration(y=y, sr=sample_rate) * 1000)

        if y.size < frame_length:
            raise RuntimeError("The selected background music is too short.")
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
        bpm, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            units="frames",
        )

        bpm_val = float(np.atleast_1d(bpm)[0])

        beat_frames = np.asarray(beat_frames, dtype=int)

        beat_times = self._compute_accent_beats(y=y, sr=sample_rate, beat_frames=beat_frames, hop_length=hop_length)

        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        energy_mean = float(np.mean(rms))

        rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
        energy_mean_db = float(np.mean(rms_db))

        lo = float(np.percentile(rms_db, 10.0))
        hi = float(np.percentile(rms_db, 95.0))
        dynamic_range_db = float(hi - lo)

        return {
            "bgm_id": bgm_info.get("id"),
            "path": str(path),
            "duration": duration,
            "sample_rate": sample_rate,
            "bpm": bpm_val,
            "beats": beat_times,
            "energy_mean": energy_mean,
            "energy_mean_db": energy_mean_db,
            "dynamic_range_db": dynamic_range_db,
        }


    @staticmethod
    def _load_audio_mono(path: Path, sr: int) -> tuple[np.ndarray, int]:

        try:
            y, sr_out = librosa.load(path, sr=sr, mono=True)
            return y.astype(np.float32, copy=False), int(sr_out)
        except Exception as e1:

            # Librosa failed to read. ffmpeg is used as a fallback
            import os
            import subprocess
            import tempfile

            tmp_wav = None
            try: 
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_wav = tmp.name
                
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", str(path),
                    "-ac", "1",
                    "-ar", str(sr),
                    "-vn",
                    tmp_wav,
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                y, sr_out = librosa.load(tmp_wav, sr=sr, mono=True)
                return y.astype(np.float32, copy=False), int(sr_out)
            
            except FileNotFoundError as e_ffmpeg:
                raise RuntimeError(
                    f"The audio cannot be loaded and ffmpeg is not found."
                ) from e_ffmpeg

            except Exception as e2:
                raise RuntimeError(
                    f"The audio cannot be loaded: {type(e1).__name__}: {e1}"
                    f"Ffmpeg error: {type(e2).__name__}: {e2}"
                ) from e2
            finally:
                if tmp_wav is not None:
                    try:
                        os.remove(tmp_wav)
                    except Exception:
                        pass


    @staticmethod
    def _compute_accent_beats(
        y: np.ndarray,
        sr: int,
        beat_frames: np.ndarray,
        hop_length: int,
        top_pct: float = 70.0,          
        min_sep_beats: int = 1,         # Min beat separation: 1 prevents selecting adjacent beats
        use_percussive: bool = True,    # Calculate onset strength from percussive component
        local_norm_win: int = 8,        # Window size for local normalization (measured in beats)
        require_local_peak: bool = True # Only retain onsets that are local maxima
    ) -> list[float]:
        """
        Calculate timestamps of the top `top_pct` percent of drum beats by intensity
        """

        if beat_frames.size == 0:
            return []

        # 1) Use percussive version for onset envelope
        y_for_onset = librosa.effects.percussive(y) if use_percussive else y
        onset_env = librosa.onset.onset_strength(y=y_for_onset, sr=sr, hop_length=hop_length)

        # 2) Use onset strength at each beat time as beat strength
        beat_frames_clip = np.clip(beat_frames.astype(int), 0, len(onset_env) - 1)
        strength = onset_env[beat_frames_clip].astype(np.float64)  # shape (n_beats,)

        # 3) Local normalization: prevent louder sections from dominating beat selection
        if strength.size >= 3 and local_norm_win >= 3:
            w = min(int(local_norm_win), int(strength.size))
            kernel = np.ones(w, dtype=np.float64) / w
            local_mean = np.convolve(strength, kernel, mode="same")
            strength_norm = strength / (local_mean + 1e-8)
        else:
            strength_norm = strength.copy()

        # 4) Select beats in the top top_pct percentile
        thr = float(np.percentile(strength_norm, 100.0 - top_pct))
        cand = np.where(strength_norm >= thr)[0]  # indices into beats

        # 5) Retain only local peaks to prevent selecting many beats during plateaus
        if require_local_peak and cand.size > 0 and strength_norm.size >= 3:
            is_peak = np.zeros_like(strength_norm, dtype=bool)
            is_peak[1:-1] = (strength_norm[1:-1] >= strength_norm[:-2]) & (strength_norm[1:-1] >= strength_norm[2:])
            is_peak[0] = strength_norm[0] >= strength_norm[1]
            is_peak[-1] = strength_norm[-1] >= strength_norm[-2]
            cand = cand[is_peak[cand]]

        # 6) Minimum separation suppression
        selected = []
        if cand.size > 0:
            order = cand[np.argsort(-strength_norm[cand])]
            suppressed = np.zeros(strength_norm.size, dtype=bool)

            for idx in order:
                if suppressed[idx]:
                    continue
                selected.append(int(idx))
                lo = max(0, idx - min_sep_beats)
                hi = min(strength_norm.size, idx + min_sep_beats + 1)
                suppressed[lo:hi] = True

        selected = np.array(sorted(selected), dtype=int)

        accent_frames = beat_frames[selected]
        accent_times = librosa.frames_to_time(accent_frames, sr=sr, hop_length=hop_length).tolist()

        accent_times_ms = [round(x * 1000) for x in accent_times]

        return accent_times_ms
