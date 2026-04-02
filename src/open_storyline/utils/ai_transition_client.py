import base64
import json
import os
import time
import requests
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Dict, Tuple


class BaseVisionClient(ABC):
    DEFAULT_DURATION = 5
    DEFAULT_RESOLUTION = "720P"

    def __init__(self, api_key: str, timeout: int = 600, cancel_checker=None):
        self.api_key = api_key
        self.timeout = timeout
        self.duration = self.DEFAULT_DURATION
        self.resolution = self.DEFAULT_RESOLUTION
        self.cancel_checker = cancel_checker
    
    def get_default_resolution(self, model: str) -> str:
        return self.DEFAULT_RESOLUTION

    @abstractmethod
    def _build_payload(
        self, 
        prompt: str, 
        model: str, 
        first_frame: str | None = None, 
        last_frame: str | None = None,
        resolution: str = "720P",
        duration: int = 5,
        prompt_optimizer: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_endpoint(self, task_type: str) -> str:
        pass

    @abstractmethod
    def _extract_task_id(self, response_json: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def check_status(self, task_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        pass

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _raise_if_cancelled(self) -> None:
        checker = self.cancel_checker
        if checker and checker():
            raise RuntimeError("generate_ai_transition cancelled by user")

    def _sleep_with_cancel(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.2, remaining))

    def _format_response_error(self, response: requests.Response, action: str) -> str:
        status_line = f"HTTP {response.status_code}"
        if response.reason:
            status_line = f"{status_line} {response.reason}"

        detail = ""
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if payload is not None:
            detail = json.dumps(payload, ensure_ascii=False)
        else:
            detail = (response.text or "").strip()

        if detail:
            detail = detail[:2000]
            return f"{self.__class__.__name__} {action} failed: {status_line}. Response: {detail}"
        return f"{self.__class__.__name__} {action} failed: {status_line}."

    def _raise_for_status_with_details(self, response: requests.Response, action: str) -> None:
        if response.ok:
            return
        raise RuntimeError(self._format_response_error(response, action))

    def submit_task(
        self, 
        task_type: str, 
        prompt: str, 
        model: str, 
        first_frame: str | None = None, 
        last_frame: str | None = None,
        resolution: str | None = None,
        duration: int | None = None,
        prompt_optimizer: bool = True,
        **kwargs
    ) -> str:
        if isinstance(resolution, str):
            resolution = resolution.strip() or None
        resolution = self.get_default_resolution(model) if resolution is None else resolution
        duration = self.duration if duration is None else duration
        url = self._get_endpoint(task_type)
        payload = self._build_payload(
            prompt, model, first_frame, last_frame, 
            resolution, duration, prompt_optimizer, **kwargs
        )
        self._raise_if_cancelled()

        try:
            response = requests.post(url, json=payload, headers=self._get_headers())
        except requests.RequestException as e:
            raise RuntimeError(f"{self.__class__.__name__} submit task request failed: {e}") from e
        self._raise_for_status_with_details(response, "submit task")
        self._raise_if_cancelled()
        
        task_id = self._extract_task_id(response.json())
        if not task_id:
            raise ValueError(f"Provider API returned invalid response: {response.text}")
        return task_id

    def poll_for_result(self, task_id: str, poll_interval: int = 15) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            self._raise_if_cancelled()
            result_url, data = self.check_status(task_id)
            if result_url:
                return result_url, data
            
            self._sleep_with_cancel(poll_interval)
            print(f"[{self.__class__.__name__}] Task {task_id} processing...")
            
        raise TimeoutError(f"Task {task_id} timed out after {self.timeout} seconds.")

    def download_asset(self, url: str, output_dir: str, task_id: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        self._raise_if_cancelled()
        try:
            response = requests.get(url, stream=True)
        except requests.RequestException as e:
            raise RuntimeError(f"{self.__class__.__name__} download asset request failed: {e}") from e
        self._raise_for_status_with_details(response, "download asset")

        content_type = response.headers.get('content-type', '')
        ext = mimetypes.guess_extension(content_type) or ".mp4"
        save_path = os.path.join(output_dir, f"result_{task_id}{ext}")
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                self._raise_if_cancelled()
                f.write(chunk)
        return save_path

    def generate(
        self, 
        task_type: str, 
        prompt: str, 
        model: str, 
        first_frame: str | None = None, 
        last_frame: str | None = None,
        resolution: str | None = None,
        duration: int | None = None,
        prompt_optimizer: bool = True,
        output_dir: str = "./output",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        task_id = self.submit_task(
            task_type, prompt, model, first_frame, last_frame, 
            resolution, duration, prompt_optimizer, **kwargs
        )
        result_url, raw_data = self.poll_for_result(task_id)
        file_path = self.download_asset(result_url, output_dir, task_id)
        return file_path, raw_data


class MiniMaxVisionClient(BaseVisionClient):
    BASE_URL = "https://api.minimaxi.com/v1"
    DEFAULT_DURATION = 6
    DEFAULT_RESOLUTION = "768P"

    MODEL_DEFAULT_RESOLUTIONS = {
        "MiniMax-Hailuo-02": "768P",
    }

    def get_default_resolution(self, model: str) -> str:
        return self.MODEL_DEFAULT_RESOLUTIONS.get(model, self.DEFAULT_RESOLUTION)

    def _get_endpoint(self, task_type: str) -> str:
        return f"{self.BASE_URL}/video_generation"

    def _build_payload(self, prompt, model, first_frame, last_frame, resolution, duration, prompt_optimizer, **kwargs):
        payload = {
            "model": model,
            "prompt": prompt,
            "first_frame_image": first_frame,
            "last_frame_image": last_frame,
            "duration": duration,
            "resolution": resolution,
            "prompt_optimizer": prompt_optimizer,
            "aigc_watermark": kwargs.get("watermark", False)
        }
        return {k: v for k, v in payload.items() if v not in [None, ""]}

    def _extract_task_id(self, response_json: Dict[str, Any]) -> str:
        return response_json.get("task_id")

    def check_status(self, task_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        url = f"{self.BASE_URL}/query/video_generation?task_id={task_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self._raise_if_cancelled()
        try:
            response = requests.get(url, headers=headers)
        except requests.RequestException as e:
            raise RuntimeError(f"{self.__class__.__name__} query task status request failed: {e}") from e
        self._raise_for_status_with_details(response, "query task status")
        data = response.json()
        
        if data.get("status") == "Success":
            file_id = data.get("file_id")
            self._raise_if_cancelled()
            try:
                retrieve_resp = requests.get(f"{self.BASE_URL}/files/retrieve?file_id={file_id}", headers=headers)
            except requests.RequestException as e:
                raise RuntimeError(f"{self.__class__.__name__} retrieve generated file request failed: {e}") from e
            self._raise_for_status_with_details(retrieve_resp, "retrieve generated file")
            return retrieve_resp.json().get("file", {}).get("download_url"), data
        elif data.get("status") == "Fail":
            raise RuntimeError(f"MiniMax Task Failed: {data}")
        return None, data


class DashScopeVisionClient(BaseVisionClient):
    BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

    DEFAULT_DURATION = 5
    DEFAULT_RESOLUTION = "720P"

    MODEL_DEFAULT_RESOLUTIONS = {
        "wan2.2-kf2v-flash": "480P",
        "wanx2.1-kf2v-plus": "720P",
    }

    def get_default_resolution(self, model: str) -> str:
        return self.MODEL_DEFAULT_RESOLUTIONS.get(model, self.DEFAULT_RESOLUTION)

    def _get_headers(self) -> Dict[str, str]:
        headers = super()._get_headers()
        headers["X-DashScope-Async"] = "enable"
        return headers

    def _get_endpoint(self, task_type: str) -> str:
        return f"{self.BASE_URL}/services/aigc/image2video/video-synthesis"

    def _build_payload(self, prompt, model, first_frame, last_frame, resolution, duration, prompt_optimizer, **kwargs):
        return {
            "model": model,
            "input": {
                "prompt": prompt,
                "first_frame_url": first_frame,
                "last_frame_url": last_frame,
                "negative_prompt": kwargs.get("negative_prompt")
            },
            "parameters": {
                "resolution": resolution,
                "duration": duration,
                "prompt_extend": prompt_optimizer,
                "watermark": kwargs.get("watermark", False),
                "seed": kwargs.get("seed")
            }
        }

    def _extract_task_id(self, response_json: Dict[str, Any]) -> str:
        return response_json.get("output", {}).get("task_id")

    def check_status(self, task_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        url = f"{self.BASE_URL}/tasks/{task_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self._raise_if_cancelled()
        try:
            response = requests.get(url, headers=headers)
        except requests.RequestException as e:
            raise RuntimeError(f"{self.__class__.__name__} query task status request failed: {e}") from e
        self._raise_for_status_with_details(response, "query task status")
        data = response.json()
        
        output = data.get("output", {})
        if output.get("task_status") == "SUCCEEDED":
            return output.get("video_url"), data
        elif output.get("task_status") in ["FAILED", "CANCELED"]:
            raise RuntimeError(f"DashScope Task Failed: {data}")
        return None, data

class VisionClientFactory:
    @staticmethod
    def create(provider: str, api_key: str, cancel_checker=None) -> BaseVisionClient:
        p = provider.lower()
        if p == "minimax":
            return MiniMaxVisionClient(api_key=api_key, cancel_checker=cancel_checker)
        elif p in ["dashscope", "aliyun"]:
            return DashScopeVisionClient(api_key=api_key, cancel_checker=cancel_checker)
        raise ValueError(f"Unsupported provider: {provider}")


def _normalize_media_input(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


if __name__ == "__main__":

    client = DashScopeVisionClient(api_key="")
    file_path, response = client.generate(
        task_type="video_generation",
        prompt="以一镜到底的方式拍摄，场景丝滑过渡",
        model="wan2.2-kf2v-flash",
        first_frame=_normalize_media_input(""),
        last_frame=_normalize_media_input(""),
        resolution="720P",
        duration=5,
        prompt_optimizer=False,
        output_dir="",
    )

    print(f"Generated video saved to: {file_path}")
    print(json.dumps(response, ensure_ascii=False, indent=2))
