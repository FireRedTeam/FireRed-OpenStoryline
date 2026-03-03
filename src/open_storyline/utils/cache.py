"""
节点缓存模块

设计说明：
1. 单进程保证正确性
   - 多进程下每个进程各自一套 cache，可能互相覆盖
   - 多进程部署建议关闭 cache 或使用分布式后端（如 Redis）

2. 初始化
   - 在 FastAPI/CLI/MCP server 入口调用 CacheManager.initialize(cfg)
   - 多次调用只会初始化一次

3. 文件名使用完整 hash
   - 避免碰撞导致错误结果

4. 性能建议
   - 高并发场景将 cache_dir 放到 SSD
   - 可通过调小 TTL 或 max_size_mb 控制磁盘占用
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from open_storyline.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheKey:
    """
    缓存键
    
    组成：
    - node_id: 节点标识
    - input_hash: 输入参数哈希（完整 hash，不截断）
    - config_version: 配置版本
    - session_id: 会话ID（可选，用于 session 隔离）
    """
    
    node_id: str
    input_hash: str
    config_version: str = "v1"
    session_id: Optional[str] = None
    
    def __str__(self) -> str:
        """index key 使用完整 hash，避免碰撞"""
        if self.session_id:
            return f"{self.session_id}:{self.node_id}:{self.config_version}:{self.input_hash}"
        return f"{self.node_id}:{self.config_version}:{self.input_hash}"
    
    def to_filename(self) -> str:
        """文件名使用完整 hash（避免碰撞）"""
        if self.session_id:
            return f"{self.session_id}_{self.node_id}_{self.config_version}_{self.input_hash}.json"
        return f"{self.node_id}_{self.config_version}_{self.input_hash}.json"


class CacheKeyBuilder:
    """
    缓存键构建器
    
    职责：根据节点定义的 cache_key_fields 计算稳定的哈希值
    """
    
    GLOBAL_EXCLUDE_FIELDS: Set[str] = {
        'timestamp', 'artifact_id', 'session_id', 'created_at',
        'request_id', 'trace_id', 'temp_dir', 'cache_dir'
    }
    
    @classmethod
    def build(
        cls,
        node_id: str,
        input_data: Dict[str, Any],
        policy: 'CachePolicy',
        cache_key_fields: Optional[List[str]] = None,
        exclude_fields: Optional[Set[str]] = None,
        session_id: Optional[str] = None
    ) -> CacheKey:
        """
        构建缓存键
        
        session_id 的处理统一在这里：
        - 如果 policy.session_isolated = True，则使用传入的 session_id
        - 否则 session_id = None（跨 session 共享）
        """
        if cache_key_fields:
            hash_data = {k: v for k, v in input_data.items() if k in cache_key_fields}
        else:
            exclude = cls.GLOBAL_EXCLUDE_FIELDS | (exclude_fields or set())
            hash_data = {k: v for k, v in input_data.items() if k not in exclude}
        
        try:
            serialized = cls._serialize(hash_data)
            data_str = json.dumps(serialized, sort_keys=True, ensure_ascii=False)
            input_hash = hashlib.sha256(data_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute cache key for {node_id}: {e}")
            input_hash = f"error-{time.time()}-{id(input_data)}"
        
        effective_session_id = session_id if policy.session_isolated else None
        
        return CacheKey(
            node_id=node_id,
            input_hash=input_hash,
            config_version=policy.config_version,
            session_id=effective_session_id
        )
    
    @classmethod
    def _serialize(cls, obj: Any) -> Any:
        """递归序列化对象"""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: cls._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._serialize(item) for item in obj]
        if isinstance(obj, set):
            return sorted([cls._serialize(item) for item in obj])
        try:
            return str(obj)
        except Exception:
            return f"<{type(obj).__name__}>"


@dataclass
class CachePolicy:
    """缓存策略配置"""
    enabled: bool = True
    max_size_mb: int = 1024
    ttl_seconds: int = 86400
    config_version: str = "v1"
    session_isolated: bool = False
    exclude_nodes: Set[str] = field(default_factory=set)
    node_limits: Dict[str, int] = field(default_factory=dict)


class NodeCacheBackend:
    """
    节点缓存后端
    
    注意：
    - 仅支持单进程使用
    - 多进程环境需使用分布式缓存（如 Redis）
    """
    
    def __init__(
        self,
        cache_dir: Path,
        policy: CachePolicy
    ):
        self.cache_dir = cache_dir
        self.policy = policy
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()
        self._reconcile_index()
    
    def _load_index(self):
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                with self.index_file.open("r", encoding="utf-8") as f:
                    self.index: Dict[str, Dict] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.index = {}
        else:
            self.index = {}
    
    def _save_index(self):
        """保存缓存索引（原子写入）"""
        try:
            temp_file = self.index_file.with_suffix(".tmp")
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
            temp_file.rename(self.index_file)
        except IOError as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _reconcile_index(self):
        """
        自愈：确保 index 与磁盘文件一致
        
        策略：保守处理，删除无法确认的文件
        - 缓存可重新生成，不冒险引入不确定状态
        """
        try:
            disk_files = set()
            for f in self.cache_dir.glob("*.json"):
                if f.name == "cache_index.json":
                    continue
                disk_files.add(f.name)
            
            to_remove = []
            for key_str, info in self.index.items():
                filename = info.get("file")
                if filename and filename not in disk_files:
                    logger.warning(f"Cache file missing, removing from index: {filename}")
                    to_remove.append(key_str)
            
            for key_str in to_remove:
                del self.index[key_str]
            
            indexed_files = {info.get("file") for info in self.index.values()}
            orphan_files = disk_files - indexed_files
            
            for filename in orphan_files:
                try:
                    (self.cache_dir / filename).unlink()
                    logger.info(f"Removed orphan cache file: {filename}")
                except IOError as e:
                    logger.warning(f"Failed to remove orphan file {filename}: {e}")
            
            if to_remove or orphan_files:
                self._save_index()
                logger.info(f"Index reconciled: removed {len(to_remove)} entries, deleted {len(orphan_files)} orphan files")
                
        except Exception as e:
            logger.error(f"Failed to reconcile index: {e}")
    
    def get(self, key: CacheKey) -> Optional[Dict[str, Any]]:
        """获取缓存"""
        if not self.policy.enabled:
            return None
        
        if key.node_id in self.policy.exclude_nodes:
            return None
        
        key_str = str(key)
        
        if key_str not in self.index:
            return None
        
        cache_info = self.index[key_str]
        cache_file = self.cache_dir / cache_info["file"]
        
        cache_time = cache_info.get("timestamp", 0)
        if time.time() - cache_time > self.policy.ttl_seconds:
            logger.debug(f"Cache expired: {key_str}")
            self._remove(key_str, cache_file)
            return None
        
        try:
            if not cache_file.exists():
                logger.warning(f"Cache file missing: {cache_file}")
                self._remove(key_str, cache_file)
                return None
            
            with cache_file.open("r", encoding="utf-8") as f:
                result = json.load(f)
            
            logger.info(f"Cache hit: {key.node_id}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Cache file corrupted: {cache_file}")
            self._remove(key_str, cache_file)
            return None
        except IOError as e:
            logger.error(f"Failed to read cache: {e}")
            return None
    
    def set(self, key: CacheKey, data: Dict[str, Any]) -> bool:
        """保存缓存"""
        if not self.policy.enabled:
            return False
        
        if key.node_id in self.policy.exclude_nodes:
            return False
        
        try:
            if not self._check_size_limit(key.node_id):
                self._cleanup()
            
            cache_filename = key.to_filename()
            cache_file = self.cache_dir / cache_filename
            temp_file = cache_file.with_suffix(".tmp")
            
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            temp_file.rename(cache_file)
            
            key_str = str(key)
            self.index[key_str] = {
                "file": cache_filename,
                "timestamp": time.time(),
                "size": cache_file.stat().st_size,
                "node_id": key.node_id
            }
            self._save_index()
            
            logger.info(f"Cache saved: {key.node_id}")
            return True
            
        except (IOError, TypeError, ValueError) as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def _check_size_limit(self, node_id: str) -> bool:
        """检查是否超过大小限制"""
        total_size = sum(info.get("size", 0) for info in self.index.values())
        
        node_limit = self.policy.node_limits.get(node_id)
        if node_limit:
            node_size = sum(
                info.get("size", 0) 
                for info in self.index.values() 
                if info.get("node_id") == node_id
            )
            if node_size > node_limit * 1024 * 1024:
                return False
        
        return total_size < self.policy.max_size_mb * 1024 * 1024
    
    def _cleanup(self):
        """清理旧缓存"""
        sorted_entries = sorted(
            self.index.items(),
            key=lambda x: x[1].get("timestamp", 0)
        )
        
        remove_count = max(1, len(sorted_entries) // 5)
        
        for key_str, cache_info in sorted_entries[:remove_count]:
            cache_file = self.cache_dir / cache_info["file"]
            self._remove(key_str, cache_file)
        
        logger.info(f"Cleaned up {remove_count} cache entries")
    
    def _remove(self, key_str: str, cache_file: Path):
        """删除缓存条目"""
        try:
            if cache_file.exists():
                cache_file.unlink()
            if key_str in self.index:
                del self.index[key_str]
                self._save_index()
        except IOError as e:
            logger.error(f"Failed to remove cache: {e}")
    
    def clear(self):
        """清空所有缓存"""
        try:
            for cache_info in self.index.values():
                cache_file = self.cache_dir / cache_info["file"]
                if cache_file.exists():
                    cache_file.unlink()
            self.index = {}
            self._save_index()
            logger.info("Cleared all cache")
        except IOError as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(info.get("size", 0) for info in self.index.values())
        node_counts = {}
        for info in self.index.values():
            node_id = info.get("node_id", "unknown")
            node_counts[node_id] = node_counts.get(node_id, 0) + 1
        
        return {
            "total_entries": len(self.index),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "node_counts": node_counts,
            "max_size_mb": self.policy.max_size_mb,
            "ttl_seconds": self.policy.ttl_seconds,
            "config_version": self.policy.config_version,
            "session_isolated": self.policy.session_isolated
        }


class CacheManager:
    """
    缓存管理器（进程级单例）
    
    职责：
    - 管理 NodeCacheBackend 实例
    - 确保每个进程只创建一个 backend
    """
    
    _instance: Optional['CacheManager'] = None
    _backend: Optional[NodeCacheBackend] = None
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()
    
    @classmethod
    def initialize(cls, cfg: Any) -> Optional[NodeCacheBackend]:
        """
        初始化缓存后端
        
        可在多个入口调用（FastAPI/CLI/MCP server），只会初始化一次
        """
        if cls._initialized:
            return cls._backend
        
        with cls._lock:
            if cls._initialized:
                return cls._backend
            
            cache_config = getattr(cfg, 'cache', None)
            if not cache_config or not cache_config.enabled:
                cls._initialized = True
                return None
            
            if cache_config.cache_dir:
                cache_dir = cache_config.cache_dir
            else:
                outputs_dir = getattr(cfg.project, 'outputs_dir', Path('./outputs'))
                cache_dir = Path(outputs_dir) / ".cache"
            
            policy = CachePolicy(
                enabled=cache_config.enabled,
                max_size_mb=cache_config.max_size_mb,
                ttl_seconds=cache_config.ttl_seconds,
                config_version=cache_config.config_version,
                session_isolated=cache_config.session_isolated,
                exclude_nodes=set(cache_config.exclude_nodes),
                node_limits=cache_config.node_limits
            )
            
            cls._backend = NodeCacheBackend(cache_dir, policy)
            cls._initialized = True
            logger.info(f"Cache backend initialized: {cache_dir}")
            
            return cls._backend
    
    @classmethod
    def get_backend(cls) -> Optional[NodeCacheBackend]:
        """获取缓存后端"""
        return cls._backend
    
    @classmethod
    def reset(cls):
        """重置（仅用于测试）"""
        with cls._lock:
            cls._backend = None
            cls._initialized = False
            cls._instance = None
