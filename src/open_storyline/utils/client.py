import os
import time
import requests
import mimetypes
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Tuple


class BaseVisionClient(ABC):
    def __init__(self, api_key: str, timeout: int = 600):
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    def _build_payload(
        self, 
        prompt: str, 
        model: str, 
        first_frame: str = None, 
        last_frame: str = None,
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

    def submit_task(
        self, 
        task_type: str, 
        prompt: str, 
        model: str, 
        first_frame: str = None, 
        last_frame: str = None,
        resolution: str = "720P",
        duration: int = 5,
        prompt_optimizer: bool = True,
        **kwargs
    ) -> str:
        url = self._get_endpoint(task_type)
        payload = self._build_payload(
            prompt, model, first_frame, last_frame, 
            resolution, duration, prompt_optimizer, **kwargs
        )
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        
        task_id = self._extract_task_id(response.json())
        if not task_id:
            raise ValueError(f"Provider API returned invalid response: {response.text}")
        return task_id

    def poll_for_result(self, task_id: str, poll_interval: int = 15) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            result_url, data = self.check_status(task_id)
            if result_url:
                return result_url, data
            
            time.sleep(poll_interval)
            print(f"[{self.__class__.__name__}] Task {task_id} processing...")
            
        raise TimeoutError(f"Task {task_id} timed out after {self.timeout} seconds.")

    def download_asset(self, url: str, output_dir: str, task_id: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        ext = mimetypes.guess_extension(content_type) or ".mp4"
        save_path = os.path.join(output_dir, f"result_{task_id}{ext}")
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path

    def generate(
        self, 
        task_type: str, 
        prompt: str, 
        model: str, 
        first_frame: str = None, 
        last_frame: str = None,
        resolution: str = "720P",
        duration: int = 5,
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
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "Success":
            file_id = data.get("file_id")
            retrieve_resp = requests.get(f"{self.BASE_URL}/files/retrieve?file_id={file_id}", headers=headers)
            retrieve_resp.raise_for_status()
            return retrieve_resp.json().get("file", {}).get("download_url"), data
        elif data.get("status") == "Fail":
            raise RuntimeError(f"MiniMax Task Failed: {data}")
        return None, data


class DashScopeVisionClient(BaseVisionClient):
    BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

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
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        output = data.get("output", {})
        if output.get("task_status") == "SUCCEEDED":
            return output.get("video_url"), data
        elif output.get("task_status") in ["FAILED", "CANCELED"]:
            raise RuntimeError(f"DashScope Task Failed: {data}")
        return None, data


class VisionClientFactory:
    @staticmethod
    def create(provider: str, api_key: str) -> BaseVisionClient:
        p = provider.lower()
        if p == "minimax":
            return MiniMaxVisionClient(api_key=api_key)
        elif p in ["dashscope", "aliyun"]:
            return DashScopeVisionClient(api_key=api_key)
        raise ValueError(f"Unsupported provider: {provider}")