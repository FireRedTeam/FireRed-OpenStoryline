import os
import requests
import time
import logging

from typing import Any, Dict, Optional, ClassVar, Type, Tuple, List
from pydantic import BaseModel

from pathlib import Path

from open_storyline.nodes.core_nodes.base_node import NodeMeta, BaseNode
from open_storyline.nodes.node_schema import SearchMediaInput
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.register import NODE_REGISTRY

logger = logging.getLogger(__name__)

SEARCH_RESULT_PER_PAGE = 40
MAX_PHOTO_NUMBER = 10
MAX_VIDEO_NUMBER = 10
MIN_VIDEO_DURATION = 1
MAX_VIDEO_DURATION = 30
DEFAULT_RESULT_NUMBER_PER_PAGE = 50
DEFAULT_PAGE = 1

TARGET_LONG_EDGE_PX = 1080

VALID_ORIENTATIONS = {"landscape", "portrait"}
VIDEO_QUALITY_RANK = {"sd": 0, "hd": 1, "uhd": 2}

MAX_RETRIES = 3
RETRY_INITIAL_DELAY = 1.0
RETRY_MAX_DELAY = 10.0
RETRY_BACKOFF_FACTOR = 2.0

@NODE_REGISTRY.register()
class SearchMediaNode(BaseNode):
    meta = NodeMeta(
        name="search_media",
        description="search",
        node_id="search_media",
        node_kind="search_media",
        require_prior_kind=[],
        default_require_prior_kind=[],
        next_available_node=['load_media'],
    )
    input_schema: ClassVar[Type[BaseModel]] = SearchMediaInput

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pexels_api_key = inputs.get("pexels_api_key", "")
        video_saved_paths = []
        image_saved_paths = []

        if pexels_api_key == "":
            pexels_api_key = self.server_cfg.search_media.pexels_api_key
        if not pexels_api_key or pexels_api_key == "":
            pexels_api_key = os.getenv("PEXELS_API_KEY")
        if not pexels_api_key or pexels_api_key == "":
            node_state.node_summary.info_for_llm("If the user has not entered their Pexels API key, please remind them to enter it in the sidebar of the webpage.")
            raise RuntimeError("Pexels api key not detected. If you use your own pexels key, please fill in the api key in the sidebar or config.toml")
        
        root_dir = os.path.abspath(os.path.expanduser(self.server_cache_dir))
        media_dir = Path(os.path.join(root_dir, node_state.session_id, "media"))
    
        search_keyword = inputs.get("search_keyword", "")
        photo_number = min(inputs.get("photo_number", MAX_PHOTO_NUMBER), MAX_PHOTO_NUMBER)
        video_number = min(inputs.get("video_number", MAX_VIDEO_NUMBER), MAX_VIDEO_NUMBER)
        orientation = inputs.get("orientation", "")
        min_video_duration = min(max(inputs.get("min_video_duration", MIN_VIDEO_DURATION), MIN_VIDEO_DURATION), MAX_VIDEO_DURATION)
        max_video_duration = max(min(inputs.get("max_video_duration", MAX_VIDEO_DURATION), MAX_VIDEO_DURATION), MIN_VIDEO_DURATION)

        if video_number > 0:
            video_preview_urls, video_saved_paths = get_video_media_from_pexels(
                pexels_api_key=pexels_api_key,
                query=search_keyword,
                media_dir=media_dir,
                video_number=video_number,
                orientation=orientation,
                min_video_duration=min_video_duration,
                max_video_duration=max_video_duration,
            )
            node_state.node_summary.info_for_user(f"search media successfully, found {len(video_preview_urls)} videos", preview_urls=video_preview_urls)

        if photo_number > 0:
            image_preview_urls, image_saved_paths = get_photo_media_from_pexels(
                pexels_api_key=pexels_api_key,
                query=search_keyword,
                media_dir=media_dir,
                photo_number=photo_number,
                orientation=orientation,
            )
            node_state.node_summary.info_for_user(f"search media successfully, found {len(image_preview_urls)} photos", preview_urls=image_preview_urls)
        return {"search_media": video_saved_paths + image_saved_paths}


def download_video(url: str, out_path: Path) -> None:
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** attempt), RETRY_MAX_DELAY)
                logger.warning(f"Download video failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"Download video failed after {MAX_RETRIES} attempts: {e}")
    raise last_exception

def search_videos(pexels_api_key: str, query: str, per_page, page) -> dict[str, Any]:
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": pexels_api_key}
    params = {"query": query, "per_page": per_page, "page": page}
    
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** attempt), RETRY_MAX_DELAY)
                logger.warning(f"Search videos failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"Search videos failed after {MAX_RETRIES} attempts: {e}")
    raise last_exception

def filter_videos(
        raw_videos: dict[str, Any],
        video_number: int,
        orientation: str,
        min_video_duration: int,
        max_video_duration: int,
    ) -> list[str]:

    if video_number <= 0:
        return []

    desired_orientation = _normalize_orientation(orientation)

    results: list[str] = []
    seen: set[str] = set()

    videos = raw_videos.get("videos") or []
    for v in videos:
        duration = int(v.get("duration", 0))

        if duration < int(min_video_duration) or duration > int(max_video_duration):
            continue

        w = v.get("width")
        h = v.get("height")
        if w is None or h is None:
            continue
        try:
            w_i = int(w)
            h_i = int(h)
        except (TypeError, ValueError):
            continue

        if desired_orientation is not None:
            actual_orientation = _infer_orientation(w_i, h_i)
            if actual_orientation != desired_orientation:
                continue

        link = _pick_best_video_link(v.get("video_files") or [])
        if not link:
            continue

        if link in seen:
            continue

        results.append(link)
        seen.add(link)

        if len(results) >= video_number:
            break

    return results

def get_video_media_from_pexels(
        pexels_api_key: str,
        query: str,
        media_dir: Path,
        video_number: int,
        orientation: str,
        min_video_duration: int,
        max_video_duration: int
    ) -> Tuple[list[str], List[Dict[str, Any]]]:

    if video_number <= 0:
        return ([], [])

    media_dir.mkdir(parents=True, exist_ok=True)

    collected: list[str] = []
    seen: set[str] = set()

    page = DEFAULT_PAGE
    while len(collected) < video_number:
        raw_videos = search_videos(
            pexels_api_key=pexels_api_key,
            query=query,
            per_page=DEFAULT_RESULT_NUMBER_PER_PAGE,
            page=page,
        )

        batch = filter_videos(
            raw_videos=raw_videos,
            video_number=video_number - len(collected),
            orientation=orientation,
            min_video_duration=min_video_duration,
            max_video_duration=max_video_duration,
        )

        for url in batch:
            if url not in seen:
                collected.append(url)
                seen.add(url)

        if not raw_videos.get("next_page") or not (raw_videos.get("videos") or []):
            break
        page += 1

    video_save_path = []
    ts = int(time.time() * 1000)
    for idx, url in enumerate(collected):
        out_path = media_dir / f"pexels_video_{ts}_{idx}.mp4"
        download_video(url, out_path)
        video_save_path.append({'path': str(out_path)})

    return collected, video_save_path

def _normalize_orientation(orientation: str) -> Optional[str]:
    normalize_orientation = (orientation or "").strip().lower()
    return normalize_orientation if normalize_orientation in VALID_ORIENTATIONS else None

def _infer_orientation(width: int, height: int) -> str:
    return "landscape" if width > height else "portrait"

def _pick_best_video_link(video_files: list[dict[str, Any]]) -> Optional[str]:
    """
    Pick a "moderate" MP4 download link.
    """
    mp4_candidates: list[dict[str, Any]] = []
    for file_info in video_files or []:
        is_mp4 = file_info.get("file_type") == "video/mp4"
        has_link = bool(file_info.get("link"))
        if is_mp4 and has_link:
            mp4_candidates.append(file_info)

    if not mp4_candidates:
        return None

    def quality_preference(quality: Any) -> int:
        # Higher is better.
        quality_str = (str(quality).lower() if quality is not None else "")
        if quality_str == "hd":
            return 2
        if quality_str == "sd":
            return 1
        if quality_str == "uhd":
            return 0
        return -1

    def candidate_score(file_info: dict[str, Any]) -> tuple[int, int, int]:
        width_px = int(file_info.get("width", 0))
        height_px = int(file_info.get("height", 0))
        file_size_bytes = int(file_info.get("size", 0))

        long_edge_px = max(width_px, height_px)
        long_edge_distance = abs(long_edge_px - TARGET_LONG_EDGE_PX)

        return (
            -long_edge_distance,
            quality_preference(file_info.get("quality")),  # hd > sd > uhd > unknown
            -file_size_bytes,                 # smaller file is better
        )

    best_candidate = max(mp4_candidates, key=candidate_score)
    return best_candidate.get("link")

def download_photo(url: str, out_path: Path) -> None:
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** attempt), RETRY_MAX_DELAY)
                logger.warning(f"Download photo failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"Download photo failed after {MAX_RETRIES} attempts: {e}")
    raise last_exception

def search_photos(pexels_api_key: str, query: str, per_page, page) -> dict[str, Any]:
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": pexels_api_key}
    params = {"query": query, "per_page": per_page, "page": page}
    
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** attempt), RETRY_MAX_DELAY)
                logger.warning(f"Search photos failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"Search photos failed after {MAX_RETRIES} attempts: {e}")
    raise last_exception

def filter_photos(
        raw_photos: dict[str, Any],
        photo_number: int,
        orientation: str,
    ) -> list[str]:

    if photo_number <= 0:
        return []

    desired_orientation = _normalize_orientation(orientation)

    results: list[str] = []
    seen: set[str] = set()

    photos = raw_photos.get("photos") or []
    for p in photos:
        w = p.get("width")
        h = p.get("height")
        if w is None or h is None:
            continue
        try:
            w_i = int(w)
            h_i = int(h)
        except (TypeError, ValueError):
            continue

        if desired_orientation is not None:
            actual_orientation = _infer_orientation(w_i, h_i)
            if actual_orientation != desired_orientation:
                continue

        src = p.get("src") or {}
        if desired_orientation is not None:
            url = src.get(desired_orientation) or src.get("original")
        else:
            url = src.get("original") or src.get("large2x") or src.get("large") or src.get("medium")

        if not url:
            continue

        if url in seen:
            continue

        results.append(url)
        seen.add(url)

        if len(results) >= photo_number:
            break

    return results

def get_photo_media_from_pexels(
        pexels_api_key: str,
        query: str,
        media_dir: Path,
        photo_number: int,
        orientation: str,
    ) -> tuple[list[str], list[str]]:

    if photo_number <= 0:
        return ([], [])

    media_dir.mkdir(parents=True, exist_ok=True)

    collected: list[str] = []
    seen: set[str] = set()

    page = DEFAULT_PAGE
    while len(collected) < photo_number:
        raw_photos = search_photos(
            pexels_api_key=pexels_api_key,
            query=query,
            per_page=DEFAULT_RESULT_NUMBER_PER_PAGE,
            page=page,
        )

        batch = filter_photos(
            raw_photos=raw_photos,
            photo_number=photo_number - len(collected),
            orientation=orientation,
        )

        for url in batch:
            if url not in seen:
                collected.append(url)
                seen.add(url)

        if not raw_photos.get("next_page") or not (raw_photos.get("photos") or []):
            break
        page += 1

    image_save_paths = []
    ts = int(time.time() * 1000)
    for idx, url in enumerate(collected):
        out_path = media_dir / f"pexels_photo_{ts}_{idx}.jpg"
        download_photo(url, out_path)
        image_save_paths.append({"path": str(out_path)})

    return collected, image_save_paths