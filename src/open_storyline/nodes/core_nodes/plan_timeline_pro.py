from typing import List, Dict, Tuple, Union, Any
import re
import random
from src.open_storyline.config import Settings
from itertools import accumulate, pairwise
from open_storyline.config import PlanTimelineProConfig
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_schema import PlanTimelineInput
from open_storyline.utils.register import NODE_REGISTRY


class TimeLine:

    def edit_meterial_timeline(
        self,
        cfg: PlanTimelineProConfig,
        node_state: NodeState,
        music: Dict,
        meterial_durations: List[int],
        tts_res: List[Dict] = None,
        texts: List[List[str]] = [],
        types: List[str] = [],
        tts_indices_map: Dict = None,
        group_indices_map: Dict = None,
        title_clip_duration: int=None,
        is_on_beats: bool=False,
        beat_type: int = 1,
        **kwargs,
    ):
        '''
        Re-edit meterial durations according to tts duration or beats.
        '''
        if kwargs.get("is_speech_rough_cut", False) or kwargs.get("is_ai_transition", False):
            return 0, meterial_durations, [1.0 for _ in meterial_durations], [0 for _ in meterial_durations]
        
        min_single_text_duration, max_text_duration = cfg.min_single_text_duration, cfg.max_text_duration
        tts_durations = [item['duration'] for item in tts_res] if tts_res else [min(min_single_text_duration * len(''.join(text)), max_text_duration) for text in texts]
        meterial_durations = [x if x > 0 else cfg.img_default_duration for x in meterial_durations]

        # edit meterials
        music_offset = 0
        if is_on_beats is False:
            if tts_res:
                new_meterial_durations, time_margins = self.edit_meterial_durations_tts(cfg, node_state, meterial_durations, tts_durations, tts_indices_map, group_indices_map)
            else:
                new_meterial_durations = meterial_durations
                time_margins = [0 for _ in range(len(meterial_durations))]
        else:
            if music:
                # get beats
                beats_timestamp = [0] + music.get('beats', [])
                beats_durations = [beats_timestamp[i+1] - beats_timestamp[i] for i in range(len(beats_timestamp)-1)] + [music['duration'] - beats_timestamp[-1]]  # calculate extra music duration
                music_offset, new_meterial_durations = self.edit_meterial_durations_beats(cfg, node_state, meterial_durations, beats_durations, tts_durations, types, tts_indices_map, title_clip_duration)
                time_margins = [0 for _ in range(len(meterial_durations))]
            else:
                new_meterial_durations = meterial_durations
                time_margins = [0 for _ in range(len(meterial_durations))]

        # edit speed 
        speeds = [1.0 if old_duration > new_duration or _type == 'img' else old_duration / new_duration for _type, old_duration, new_duration in zip(types, meterial_durations, new_meterial_durations)]
        return music_offset, new_meterial_durations, speeds, time_margins

    def edit_meterial_durations_tts(
        self, 
        cfg: PlanTimelineProConfig, 
        node_state: NodeState,
        meterial_durations: List[int], 
        tts_durations: List[int], 
        tts_indices_map: Dict,
        group_indices_map: Dict = None,
    ) -> List[int]:
        '''
        Only add tts without beats.
        '''
        new_meterial_durations = []
        tts_paragraph = list(accumulate([v for _, v in tts_indices_map.items()]))
        group_paragraph = set(accumulate([v for _, v in group_indices_map.items()]))
        is_end_tts_paragraph = [paragraph in group_paragraph for paragraph in tts_paragraph]
        group_margin_proposal = random.randint(cfg.min_group_margin, cfg.max_group_margin)
        extra_margin = [group_margin_proposal if is_end is True else 0 for is_end in is_end_tts_paragraph]
        time_margins = [self.time_margin(cfg) + extra_margin[i] for i in range(len(tts_durations))]
        print(f"time_margins: {time_margins}, extra_margin: {extra_margin}")

        paragraph = [0] + list(accumulate(tts_indices_map.values()))
        for i, tts_duration in enumerate(tts_durations):
            meterial_paragraph_durations = [meterial_durations[idx] for idx in range(paragraph[i], paragraph[i+1])]
            # meterial_paragraph_durations_rate = [duration / sum(meterial_paragraph_durations) for duration in meterial_paragraph_durations]
            meterial_paragraph_durations_rate = [1 / len(meterial_paragraph_durations) for _ in meterial_paragraph_durations]

            # strategy-1: weighted duration
            new_meterial_durations += [max(int(tts_duration + time_margins[i]) * meterial_paragraph_durations_rate[j], cfg.min_clip_duration) for j in range(tts_indices_map[i])]

        return new_meterial_durations, time_margins

    def edit_meterial_durations_beats(
        self, 
        cfg: PlanTimelineProConfig,
        node_state: NodeState,
        meterial_durations: List[int], 
        beats_durations: List[int], 
        tts_durations: List[int],
        types: List[str],
        tts_indices_map: Dict,
        title_clip_duration: int=None,
    ) -> List[int]:

        new_meterial_durations = []
        beat_index = next((i for i, acc_duration in enumerate(accumulate(beats_durations)) if acc_duration >= title_clip_duration), len(beats_durations) - 1) + 1 if title_clip_duration else 0
        music_offset = sum(beats_durations[:beat_index]) - title_clip_duration if title_clip_duration else 0
        assert music_offset >= 0
        duration_rates = [round(1 / num, 2) for _, num in tts_indices_map.items() for _ in range(num)]
        temp_tts_durations = [val for val, count in zip(tts_durations, list(tts_indices_map.values())) for _ in range(count)]

        init_duration = 0
        wo_got_beats_clips = []
        min_clip_duration = cfg.min_clip_duration
        for i in range(len(meterial_durations)):

            minimum_duration = min_clip_duration if not tts_durations or temp_tts_durations[i] is None or temp_tts_durations[i]==0  else max(temp_tts_durations[i] * duration_rates[i], min_clip_duration)

            durations = init_duration
            # assert the music is enough long
            if beat_index >= len(beats_durations):
                beat_index = 0
                node_state.node_summary.add_warning("The music is not enough long. Set the music cycling.")

            while True:

                durations += beats_durations[beat_index]
                sub_durations = durations - meterial_durations[i]  # diff

                # cut video
                if sub_durations > 0:
                    durations -= beats_durations[beat_index]
                    if durations < minimum_duration:
                        while durations < minimum_duration:
                            durations += beats_durations[beat_index]
                            beat_index += 1
                            # assert the music is enough long
                            if beat_index >= len(beats_durations):
                                beat_index = 0
                                node_state.node_summary.add_warning("The music is not enough long. Set the music cycling.")
                        if types[i] == 'video':
                            wo_got_beats_clips.append(str(i))
                            init_duration = durations - max(meterial_durations[i], minimum_duration)
                            durations = max(meterial_durations[i], minimum_duration) # set new duration to max(meterial_durations[i], minimum_duration)
                        else: # img type set to the next beats.
                            init_duration = 0
                    else:
                        init_duration = 0
                    break
                else:
                    beat_index += 1

                # assert the music is enough long
                if beat_index >= len(beats_durations):
                    beat_index = 0
                    node_state.node_summary.add_warning("The music is not enough long. Set the music cycling.")

            new_meterial_durations.append(durations)

        node_state.node_summary.info_for_llm(f"[W/O. Beats Rate] {len(wo_got_beats_clips) / len(new_meterial_durations):.2f}")
        return music_offset, new_meterial_durations


    def time_margin(self, cfg: PlanTimelineProConfig):
        mode, min_time_margin, max_time_margin = cfg.tts_margin_mode, cfg.min_tts_margin, cfg.max_tts_margin
        if mode == "random":
            return random.randint(min_time_margin, max_time_margin) 
        elif mode == "avg":
            return (max_time_margin + min_time_margin) // 2
        elif mode == "min":
            return min_time_margin
        elif mode == "max":
            return max_time_margin
    
    def text_tts_offset(self, cfg: PlanTimelineProConfig):
        mode, min_text_tts_offset, max_text_tts_offset = cfg.text_tts_offset_mode, cfg.min_text_tts_offset, cfg.max_text_tts_offset
        if mode == "random":
            return random.randint(min_text_tts_offset, max_text_tts_offset) 
        elif mode == "avg":
            return (max_text_tts_offset + min_text_tts_offset) // 2
        elif mode == "min":
            return min_text_tts_offset
        elif mode == "max":
            return max_text_tts_offset

    def edit_tts_timeline(
        self,
        cfg: PlanTimelineProConfig,
        node_state: NodeState,
        meterial_durations: List[int],
        tts_res: List[Dict] = None,
        tts_indices_map: Dict = None,
        **kwargs,
    ):
        "Add tts start timestamp"
        if not tts_res or kwargs.get("is_speech_rough_cut", False) is True:
            return

        # get base start timestamps
        paragraph = [0] + list(accumulate(tts_indices_map.values()))
        paragraph_durations = [[dura for dura in meterial_durations[paragraph[i]: paragraph[i+1]]] for i in range(len(paragraph[:-1]))]
        paragraph_durations_sum = [sum(durations) for durations in paragraph_durations]
        start_timestamps = [sum(meterial_durations[:i]) for i in paragraph[:-1]]
        assert len(paragraph_durations_sum) == len(start_timestamps)

        # adjust start timestamps
        long_short_text_duration, long_text_margin_rate, short_text_margin_rate = cfg.long_short_text_duration, cfg.long_text_margin_rate, cfg.short_text_margin_rate

        long_tts_margin = [min(int(long_text_margin_rate * paragraph_durations[i][0]), abs(paragraph_durations_sum[i] - tts_res[i]['duration'])) for i in range(len(paragraph[:-1]))]
        short_tts_margin = [min(int(short_text_margin_rate * paragraph_durations[i][0]), abs(paragraph_durations_sum[i] - tts_res[i]['duration'])) for i in range(len(paragraph[:-1]))]

        start_timestamps = [start_timestamps[i] + long_tts_margin[i] if paragraph_durations_sum[i] > long_short_text_duration else start_timestamps[i] + short_tts_margin[i] for i in range(len(paragraph[:-1]))]
        
        # update to tts res
        tts_res = [{**item, 'start_timestamp': start_timestamp} for item, start_timestamp in zip(tts_res, start_timestamps)]
        return tts_res

    def edit_text_timeline(
        self,
        cfg: PlanTimelineProConfig,
        node_state: NodeState,
        meterial_durations: List[int],
        texts: List[List[str]] = "",
        tts_res: List[Dict] = None,
        tts_indices_map: Dict = None,
        music: Dict=None,
        beat_type: int = 2,
        clip_uuids: List=[],
        **kwargs,
    ): 
        '''
        Get text start timestamps and durations according to the tts and meterial durations
        '''
        text_tts_offset = [0] * len(texts)
        
        # case-0: speech rough cut, directly use the start timestamps and durations from `speech_rough_cut` output without further adjustment.
        if kwargs.get("is_speech_rough_cut", False) is True:

            node_state.node_summary.info_for_llm(f"Since the input clips are from speech rough cut, directly use the start timestamps and durations from `speech_rough_cut` output without further adjustment.")
            texts, final_text_durations, final_text_start_timestamps, text_clip_maps = [], [], [], []
            speech_rough_cut = kwargs.get("speech_rough_cut", {})
            
            for rough_cut_json in speech_rough_cut.get("rough_cut_jsons", []):
                group_texts, group_start_timestamps, group_durations = [], [], []
                for clip in rough_cut_json:
                    
                    text = re.sub(r"[^\w\s]", " ", clip.get("text", ""))
                    duration = clip.get("end", 0) - clip.get("start", 0)
                    start_timestamp = clip.get("start", 0)

                    group_texts.append(text)
                    group_start_timestamps.append(start_timestamp)
                    group_durations.append(duration)

                texts.append(group_texts)
                final_text_start_timestamps.append(group_start_timestamps)
                final_text_durations.append(group_durations)
            
            return texts, final_text_durations, final_text_start_timestamps, text_clip_maps

        # case-1: with tts
        if tts_res:

            # get tts start timestamps
            tts_start_timestamps = [item['start_timestamp'] for item in tts_res]
            tts_durations = [item['duration'] for item in tts_res]

            # offset duration
            text_tts_offset = [self.text_tts_offset(cfg) for _ in tts_res]

            # calculate text start timestamps
            text_start_timestamps = [tts_start_timestamp + offset for tts_start_timestamp, offset in zip(tts_start_timestamps, text_tts_offset)]
                        
            # calculate text durations
            base_text_durations = [b - a for a, b in pairwise(text_start_timestamps + [sum(meterial_durations)])]
            if cfg.text_duration_mode == 'with_tts':
                text_durations = [tts_duration for tts_duration in tts_durations]
            elif cfg.text_duration_mode == 'with_clip':

                # calculate tts margin
                long_short_text_duration, long_text_margin_rate, short_text_margin_rate = cfg.long_short_text_duration, cfg.long_text_margin_rate, cfg.short_text_margin_rate
                long_tts_margin = [min(int(long_text_margin_rate * paragraph_durations[i][0]), abs(paragraph_durations_sum[i] - tts_res[i]['duration'])) for i in range(len(paragraph[:-1]))]
                short_tts_margin = [min(int(short_text_margin_rate * paragraph_durations[i][0]), abs(paragraph_durations_sum[i] - tts_res[i]['duration'])) for i in range(len(paragraph[:-1]))]

                text_durations = [base_text_duration - offset for base_text_duration, offset in zip(base_text_durations, text_tts_offset)]
                text_durations = [text_duration - long_tts_margin[i] if paragraph_durations_sum[i] > long_short_text_duration else text_duration - short_tts_margin[i] for i, text_duration in enumerate(text_durations)]
            else:
                node_state.node_summary.add_warning(f"[{self.__class__.__name__}] text_duration_mode: {cfg.text_duration_mode} not in [`with_tts`, `with_clip`], return `with_tts` result as default.")
                text_durations = [tts_duration for tts_duration in tts_durations]
        
        # case-2: wo-tts, `text_duration_mode` default is `with_clip`
        else:
            # get base start timestamps
            paragraph = [0] + list(accumulate(tts_indices_map.values()))
            paragraph_durations = [[dura for dura in meterial_durations[paragraph[i]: paragraph[i+1]]] for i in range(len(paragraph[:-1]))]
            paragraph_durations_sum = [sum(durations) for durations in paragraph_durations]
            text_start_timestamps = [sum(meterial_durations[:i]) for i in paragraph[:-1]]
            text_durations = [b - a for a, b in pairwise(text_start_timestamps + [sum(meterial_durations)])]  # `text_duration_mode` default is `with_clip`
            assert len(paragraph_durations) == len(text_start_timestamps)

            # adjust start timestamps
            long_short_text_duration, long_text_margin_rate, short_text_margin_rate = cfg.long_short_text_duration, cfg.long_text_margin_rate, cfg.short_text_margin_rate

            long_tts_margin = [int(long_text_margin_rate * paragraph_durations[i][0]) for i in range(len(paragraph[:-1]))]
            short_tts_margin = [int(short_text_margin_rate * paragraph_durations[i][0]) for i in range(len(paragraph[:-1]))]

            text_start_timestamps = [text_start_timestamps[i] + long_tts_margin[i] if paragraph_durations_sum[i] > long_short_text_duration else text_start_timestamps[i] + short_tts_margin[i] for i in range(len(paragraph[:-1]))]
            # adjust tts_margin by beats
            if cfg.is_text_beats:
                beats_timestamp = [0] + music.get('beats', [])
                beats_timestamp += [music['duration'] + timestamp for timestamp in beats_timestamp]
                text_start_timestamps = self.replace_with_closest_if_within_threshold(text_start_timestamps, beats_timestamp)
                    
            # calculate text durations
            text_durations = [text_duration - long_tts_margin[i] if paragraph_durations_sum[i] > long_short_text_duration else text_duration - short_tts_margin[i] for i, text_duration in enumerate(text_durations)]
        
        # split text to sub-text
        final_text_durations, final_text_start_timestamps, text_clip_maps = [], [], []
        for text, duration, start_timestamp, offset in zip(texts, text_durations, text_start_timestamps, text_tts_offset):
            
            # obtain final start-timestamps and durations
            sub_text_durations = [int(len(sub_text) / len(''.join(text)) * duration) for sub_text in text]
            sub_start_timestamps = [start_timestamp + sum(sub_text_durations[:i]) for i in range(len(sub_text_durations))]
            final_text_durations.append(sub_text_durations)
            final_text_start_timestamps.append(sub_start_timestamps)
                
        return texts, final_text_durations, final_text_start_timestamps, text_clip_maps

    @staticmethod
    def replace_with_closest_if_within_threshold(source_list, reference_list, threshold: int=500):
        result = []
        for num in source_list:
            
            closest = min(reference_list, key=lambda x: abs(x - num))
            
            if abs(closest - num) < threshold:
                result.append(closest)
            else:
                result.append(num)
        return result


@NODE_REGISTRY.register()
class PlanTimelineProNode(BaseNode):

    meta = NodeMeta(
        name="plan_timeline_pro",
        description=(
            "Create a coherent timeline by arranging video clips, subtitles, voice-over, and background music. "
        ),
        node_id="plan_timeline_pro",
        node_kind="plan_timeline",
        require_prior_kind=["split_shots", "speech_rough_cut", "group_clips", "generate_ai_transition", "generate_script", "tts", "music_rec"],
        default_require_prior_kind=["split_shots", "group_clips", "generate_ai_transition", "generate_script", "tts", "music_rec"],
        next_available_node=["render_video"],
    )
    
    input_schema = PlanTimelineInput


    def __init__(self, server_cfg: Settings) -> None:
        super().__init__(server_cfg)
        self.default_timeline_cfg: PlanTimelineProConfig = self.server_cfg.plan_timeline_pro
        self.timeline_client = TimeLine()

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        return await self.process(node_state, inputs)

    async def process(self, node_state: NodeState,  inputs: Dict[str, Any]) -> Any:
        music = inputs.pop("music", None)
        tts_res = inputs.pop("tts_res", None)
        is_speech_rough_cut = inputs.get("is_speech_rough_cut", False)
        is_ai_transition = inputs.get("is_ai_transition", False)

        # Processing clip durations
        music_offset, new_meterial_durations, speeds, time_margins = self.timeline_client.edit_meterial_timeline(
            self.default_timeline_cfg, 
            node_state,
            music, 
            inputs.get('clip_durations'), 
            tts_res, 
            texts=inputs.get('texts', []), 
            types=inputs.get('types', []), 
            tts_indices_map=inputs.get('text_indices_map', {}), 
            group_indices_map=inputs.get('text_indices_map', {}), 
            title_clip_duration=inputs.get('title_clip_duration', 0),
            is_on_beats=inputs.get('is_on_beats', False),
            is_speech_rough_cut=is_speech_rough_cut,
            is_ai_transition=is_ai_transition,
        )

        # Processing tts durations
        tts_res = self.timeline_client.edit_tts_timeline(
            self.default_timeline_cfg, 
            node_state,
            new_meterial_durations, 
            tts_res, 
            tts_indices_map=inputs.get('text_indices_map', {}), 
            is_speech_rough_cut=is_speech_rough_cut,
        )
        tts_start_timestamps = [item.get("start_timestamp") for item in tts_res] if tts_res else []

        # Processing text durations
        texts, text_durations, text_start_timestamps, text_clip_maps = self.timeline_client.edit_text_timeline(
            self.default_timeline_cfg, 
            node_state,
            new_meterial_durations, 
            texts=inputs.get('texts', []), 
            tts_res=tts_res, 
            tts_indices_map=inputs.get('text_indices_map', {}),
            music=music, 
            clip_uuids=[],
            is_speech_rough_cut=is_speech_rough_cut,
            speech_rough_cut=inputs.get("speech_rough_cut"),
        )
        
        inputs.update({
            "music": music,
            "tts_res": tts_res,
            "texts": texts,
        })

        return {
            'timeline_source_data': inputs,
            'music_offset': music_offset,
            'new_meterial_durations': new_meterial_durations,
            'speeds': speeds,
            'time_margins': time_margins,
            'text_start_timestamps': text_start_timestamps,
            'text_durations': text_durations,
            'text_clip_maps': text_clip_maps,
            'tts_start_timestamps': tts_start_timestamps,
        }

    def _combine_tool_outputs(self, node_state, outputs):
        """
        Change output format.
        """
        tracks, video, subtitles, voiceover, bgm = [], [], [], [], []
        timeline_source_data = outputs.get('timeline_source_data', {})
        
        # Video track
        clip_ids = timeline_source_data.get('clip_ids', [])
        clip_group_ids = timeline_source_data.get('clip_group_ids', [])
        kinds = timeline_source_data.get('types', [])
        fps = timeline_source_data.get('fps', [])
        sizes = timeline_source_data.get('sizes', [])
        source_paths = timeline_source_data.get('clips', [])
        clip_durations = timeline_source_data.get('clip_durations', [])
        clip_durations = [x if x > 0 else self.default_timeline_cfg.img_default_duration for x in clip_durations]
        new_meterial_durations = outputs.get('new_meterial_durations', {})
        playback_rates = outputs.get('speeds', [])

        timeline_start = 0
        for clip_id, clip_group_id, kind, _fps, source_path, clip_duration, new_meterial_duration, playback_rate, size in \
            zip(clip_ids, clip_group_ids, kinds, fps, source_paths, clip_durations, new_meterial_durations, playback_rates, sizes):
            video.append({
                "clip_id": clip_id,
                "group_id": clip_group_id,
                "kind": kind,
                "fps": _fps,
                "size": size,
                "source_path": source_path,
                "source_window": {
                    "start": 0,
                    "end": min(clip_duration, new_meterial_duration),
                    "duration": min(clip_duration, new_meterial_duration)
                },
                "timeline_window": {
                    "start": timeline_start,
                    "end": timeline_start + new_meterial_duration,
                    "duration": new_meterial_duration
                },
                "playback_rate": playback_rate
            })
            timeline_start += new_meterial_duration
        
        # Subtitles track
        if timeline_source_data.get('texts'):
            texts = [x for item in timeline_source_data.get('texts', []) for x in item]
            text_start_timestamps = [st for item in outputs.get('text_start_timestamps', []) for st in item]
            text_durations = [t for item in outputs.get('text_durations', []) for t in item]
            text_group_ids = timeline_source_data.get('text_group_ids', []) or [None for _ in range(len(texts))]
            text_unit_ids = timeline_source_data.get('text_unit_ids', []) or [None for _ in range(len(texts))]
            text_index_in_group = timeline_source_data.get('text_index_in_group', []) or [None for _ in range(len(texts))]

            for text_group_id, text_unit_id, index_in_group, text, text_start_timestamp, text_duration in \
                zip(text_group_ids, text_unit_ids, text_index_in_group, texts, text_start_timestamps, text_durations):
                subtitles.append({
                    "group_id": text_group_id,
                    "unit_id": text_unit_id,
                    "index_in_group": index_in_group,
                    "text": text,
                    "timeline_window": {
                        "start": text_start_timestamp,
                        "end": text_start_timestamp + text_duration
                    }
                })
        
        # Voiceover track
        if timeline_source_data.get('tts_res'):
            tts_group_ids = timeline_source_data.get('tts_group_ids', [])
            voiceover_ids = timeline_source_data.get('voiceover_ids', [])
            tts_durations = timeline_source_data.get('tts_durations', [])
            tts_paths = timeline_source_data.get('tts_paths', [])
            tts_start_timestamps = outputs.get('tts_start_timestamps', [])
        
            for tts_group_id, voiceover_id, tts_duration, tts_start_timestamp, tts_path in \
                zip(tts_group_ids, voiceover_ids, tts_durations, tts_start_timestamps, tts_paths):
                voiceover.append({
                    "group_id": tts_group_id,
                    "voiceover_id": voiceover_id,
                    "source_window": {
                        "start": 0,
                        "end": tts_duration,
                        "duration": tts_duration
                    },
                    "timeline_window": {
                        "start": tts_start_timestamp,
                        "end": tts_start_timestamp + tts_duration,
                        "duration": tts_duration
                    },
                    "path": tts_path
                })
        
        # Bgm track
        if timeline_source_data.get('music'):
            music_info = timeline_source_data.get('music', {})
            music_duration = music_info.get("duration", 0)
            video_duration = int(sum(new_meterial_durations))
            loop_num = video_duration // music_duration
            for i in range(loop_num + 1):
                bgm.append({
                    "bgm_id": music_info.get("bgm_id", ""),
                    "source_window": {
                        "start": 0,
                        "end": music_duration if i != loop_num else video_duration - loop_num * music_duration
                    },
                    "path": music_info.get("path", 0)
                })
        
        # Merge all tracks
        tracks = {
            "video": video,
            "subtitles": subtitles,
            "voiceover": voiceover,
            "bgm": bgm,
        }
        return {"tracks": tracks}
    
    def _parse_input(self, node_state: NodeState, inputs, **kwargs):
        
        split_shots = inputs.get("split_shots", {})
        group_clips = inputs.get("group_clips", {})
        generate_ai_transition = inputs.get("generate_ai_transition", {})
        generate_script = inputs.get("generate_script", {})
        music = (inputs.get("music_rec") or {}).get("bgm", {})
        tts_res = inputs.get("tts", {}).get("voiceover", [])
        use_beats = inputs.get("use_beats", False)
        is_speech_rough_cut = inputs.get("is_speech_rough_cut", False)
        is_ai_transition = inputs.get("is_ai_transition", False)
        speech_rough_cut = inputs.get("speech_rough_cut")
        texts, types = [], []
        clips, clip_ids, clip_idxes = [], [], []
        clip_group_ids = []
        clip_durations = []
        text_group_ids, text_unit_ids, text_index_in_group = [], [], []
        text_indices_map = {}
        tts_group_ids, voiceover_ids, tts_durations, tts_paths = [], [], [], []

        if is_ai_transition is True:
            transition_info = generate_ai_transition.get("transition_info", {})
            groups = generate_ai_transition.get("groups", [])
            image_duration_ms = int(inputs.get("image_duration_ms", 3000) or 3000)
            start_times, fps, sizes = [], [], []
            clips_by_clip_id = {clip.get("clip_id", ""): clip for clip in split_shots.get("clips", [])}

            for group in groups:
                group_id = group.get("group_id", "")
                for clip_id in group.get("clip_ids", []) or []:
                    if isinstance(clip_id, str) and clip_id.startswith("transition_"):
                        transition_clip = transition_info.get(clip_id, {})
                        transition_source_ref = transition_clip.get("source_ref", {})
                        transition_duration = int(transition_source_ref.get("duration_ms", 0) or 0)
                        if transition_duration <= 0:
                            continue

                        clip_ids.append(clip_id)
                        clip_group_ids.append(group_id)
                        clips.append(transition_clip.get("path", ""))
                        types.append("video")
                        clip_durations.append(transition_duration)
                        start_times.append(0)
                        fps.append(transition_clip.get("fps", 0))
                        sizes.append([
                            transition_source_ref.get("width", 576),
                            transition_source_ref.get("height", 1024),
                        ])
                        continue

                    clip_info = clips_by_clip_id.get(clip_id, {})
                    if not clip_info:
                        continue

                    clip_source_ref = clip_info.get("source_ref", {})
                    clip_kind = clip_info.get("kind", "")
                    clip_duration = image_duration_ms if clip_kind == "image" else int(clip_source_ref.get("duration", 0) or 0)
                    if clip_duration <= 0:
                        continue

                    clip_ids.append(clip_id)
                    clip_group_ids.append(group_id)
                    clips.append(clip_info.get("path", ""))
                    types.append(clip_kind)
                    clip_durations.append(clip_duration)
                    start_times.append(0 if clip_kind == "image" else int(clip_source_ref.get("start", 0) or 0))
                    fps.append(clip_info.get("fps", None))
                    sizes.append([
                        clip_source_ref.get("width", 576),
                        clip_source_ref.get("height", 1024),
                    ])

            return {
                'is_ai_transition': True,
                'types': types,
                'texts': [],
                'text_unit_ids': [],
                'text_group_ids': [],
                'text_index_in_group': [],
                'clips': clips,
                'clip_ids': clip_ids,
                'clip_group_ids': clip_group_ids,
                'fps': fps,
                'sizes': sizes,
                'clip_durations': clip_durations,
                'start_times': start_times,
                'text_indices_map': {},
                'music': music,
                'tts_res': [],
                'tts_group_ids': [],
                'voiceover_ids': [],
                'tts_durations': [],
                'tts_paths': [],
                'is_on_beats': False,
                'is_speech_rough_cut': False,
                'speech_rough_cut': None,
                'title_clip_duration': 0,
            }

        if is_speech_rough_cut is True and speech_rough_cut is None:
            node_state.node_summary.add_error("The input clips are from speech rough cut, but no clips info is found in the input. Check whether the prior node `speech_rough_cut` output is correct.")
            raise ValueError("The input clips are from speech rough cut, but no clips info is found in the input. Check whether the prior node `speech_rough_cut` output is correct.")

        # Get clips and duration
        if is_speech_rough_cut is False:
            groups = group_clips.get("groups", [])
            for i, group in enumerate(groups):
                group_clip_ids = group.get('clip_ids', [])
                clip_ids += group_clip_ids
                clip_idxes += [int(item.split('_')[-1]) for item in group_clip_ids]
                clip_group_ids += [group.get('group_id') for _ in group_clip_ids]
                text_indices_map[i] = len(group.get('clip_ids', []))

            clip_durations = [split_shots.get('clips', [])[idx-1].get('source_ref', {}).get('duration', 0) for idx in clip_idxes]
            start_times = [split_shots.get('clips', [])[idx-1].get('source_ref', {}).get('start', 0) for idx in clip_idxes]
            clips = [split_shots.get('clips', [])[idx-1].get('path', '') for idx in clip_idxes]
            types = [split_shots.get('clips', [])[idx-1].get('kind', '') for idx in clip_idxes]
            fps = [split_shots.get('clips', [])[idx-1].get('fps', None) for idx in clip_idxes]
            # For save
            sizes = [[split_shots.get('clips', [])[idx-1].get('source_ref', {}).get('width', 576), split_shots.get('clips', [])[idx-1].get('source_ref', {}).get('height', 1024)] for idx in clip_idxes]
        else:
            speech_rough_cut_clips = speech_rough_cut.get("clips", [])
            clip_ids = [clip.get("clip_id", "") for clip in speech_rough_cut_clips]
            clip_idxes = [int(clip_id.split('_')[-1]) for clip_id in clip_ids]
            clip_group_ids = [clip.get("group_id", "") for clip in speech_rough_cut_clips]
            clip_durations = [clip.get('source_ref', {}).get("duration", 0) for clip in speech_rough_cut_clips]
            start_times = [clip.get('source_ref', {}).get("start", 0) for clip in speech_rough_cut_clips]
            clips = [clip.get("path", "") for clip in speech_rough_cut_clips]
            types = [clip.get("kind", "") for clip in speech_rough_cut_clips]
            fps = [clip.get('fps', None) for clip in speech_rough_cut_clips]
            sizes = [[clip.get('source_ref', {}).get('width', 576), clip.get('source_ref', {}).get('height', 1024)] for clip in speech_rough_cut_clips]
        
        # Get text info
        for item in generate_script.get('group_scripts', []):
            texts.append([sub_item.get('text', '') for sub_item in item.get('subtitle_units', [])])
            text_unit_ids += [sub_item.get('unit_id', '') for sub_item in item.get('subtitle_units', [])]
            text_group_ids += [item.get('group_id', '') for _ in item.get('subtitle_units', [])]
            text_index_in_group += [i for i in range(len(item.get('subtitle_units', [])))]

        # Get tts info
        for item in tts_res:
            tts_group_ids.append(item.get('group_id', ''))
            voiceover_ids.append(item.get('voiceover_id', ''))
            tts_durations.append(item.get('duration', ''))
            tts_paths.append(item.get('path', ''))

        return {
            'types': types,
            'texts': texts,
            'text_unit_ids': text_unit_ids,
            'text_group_ids': text_group_ids,
            'text_index_in_group': text_index_in_group,
            'clips': clips,
            'clip_ids': clip_ids,
            'clip_group_ids': clip_group_ids,
            'fps': fps,
            'sizes': sizes,
            'clip_durations': clip_durations,
            'start_times': start_times,
            'text_indices_map': text_indices_map,
            'music': music,
            'tts_res': tts_res,
            'tts_group_ids': tts_group_ids,
            'voiceover_ids': voiceover_ids,
            'tts_durations': tts_durations,
            'tts_paths': tts_paths,
            'is_on_beats': use_beats,
            'is_speech_rough_cut': is_speech_rough_cut,
            'speech_rough_cut': speech_rough_cut,
            'title_clip_duration': 0,
        }
