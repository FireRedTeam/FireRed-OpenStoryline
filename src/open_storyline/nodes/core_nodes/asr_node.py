from typing import Any, Dict
import os
import subprocess
import tempfile

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import LocalASRInput
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class LocalASRNode(BaseNode):

    meta = NodeMeta(
        name="local_asr",
        description="Filter clips based on their descriptions according to user requirements. Depends on the results from the understand_clips tool",
        node_id="local_asr",
        node_kind="asr",
        require_prior_kind=['split_shots'],
        default_require_prior_kind=['split_shots'],
        next_available_node=['group_clips'],
    )

    input_schema = LocalASRInput

    def _load_asr_model(self):

        if hasattr(self, "asr_model"):
            return self.asr_model
        else:
            from funasr import AutoModel

            self.asr_model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                vad_kwargs={"max_single_segment_time": 30000},
            )
            return self.asr_model
        
    def extract_audio_wav(self, video_path: str, tmpdir: str):
        # 1. 判断是否有音轨
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            video_path
        ]

        result = subprocess.run(probe_cmd, capture_output=True, text=True)

        if not result.stdout.strip():
            return None
        
        out_wav = os.path.join(tmpdir, "audio.wav")

        # 3. 提取音频
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            out_wav
        ]

        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return out_wav

    async def default_process(
        self,
        node_state,
        inputs: Dict[str, Any],
    ) -> Any:
        return {}

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        
        clips = inputs["split_shots"].get('clips', [])
        asr_model = self._load_asr_model()

        asr_infos = []
        for clip in clips:
            video_path = clip["path"]
            kind = clip["kind"]

            # only process video clips, for other kinds of clips, directly return empty asr text
            if kind != "video":
                asr_infos.append({
                    "clip_id": clip["clip_id"],
                    "path": video_path,
                    "kind": kind,
                    "asr_res": {},
                })
                continue
            
            with tempfile.TemporaryDirectory() as tmpdir:
                
                # extract audio wav from video clip, if no audio track, directly return empty asr text
                audio_wav = self.extract_audio_wav(video_path, tmpdir)
                if audio_wav is None:
                    asr_infos.append({
                        "clip_id": clip["clip_id"],
                        "path": video_path,
                        "kind": kind,
                        "asr_res": {},
                    })
                    node_state.node_summary.info_for_llm(f"Clip {clip['clip_id']} has no audio track, skipped for asr.")
                    continue
                
                # perform asr and get asr text, here we directly use the audio wav path as input for asr model, 
                # since funasr can support audio file input and will handle the audio loading and feature extraction internally, 
                # which can avoid the potential audio loading and feature extraction issues in different environments
                res = asr_model.generate(
                    input=audio_wav, 
                    sentence_timestamp=True
                )
                asr_infos.append({
                    "clip_id": clip["clip_id"],
                    "path": video_path,
                    "kind": kind,
                    "asr_res": res[0] if res else {},
                })

        return {
            "asr_infos": asr_infos,
        }
    
    def _combine_tool_outputs(self, node_state, outputs):
        
        asr_infos = outputs.get("asr_infos", [])
        regularized_asr_infos = []

        for asr_info in asr_infos:
            clip_id = asr_info["clip_id"]
            kind = asr_info["kind"]
            asr_res = asr_info.get("asr_res", {})

            regularized_asr_infos.append({
                "clip_id": clip_id,
                "kind": kind,
                "asr_text": asr_res.get("text", "") if asr_res else "",
                "asr_timestamps": asr_res.get("timestamp", []) if asr_res else [],
                "asr_sentence_info": asr_res.get("sentence_info", []) if asr_res else [],
            })
        return {
            "asr_infos": regularized_asr_infos,
        }
