import os
import json
import time
import hashlib
import asyncio
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir=".cache", ttl=3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self._memory_cache = {}
        self._lock = asyncio.Lock()
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_hash(self, key):
        return hashlib.md5(str(key).encode()).hexdigest()

    async def get(self, key, category="default"):
        h = self._get_hash(f"{category}:{key}")
        
        # Memory check
        async with self._lock:
            if h in self._memory_cache:
                entry = self._memory_cache[h]
                if time.time() - entry["ts"] < self.ttl:
                    return entry["data"]
                else:
                    del self._memory_cache[h]

        # Disk check
        path = os.path.join(self.cache_dir, category, f"{h}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                    if time.time() - entry["ts"] < self.ttl:
                        # Backfill memory
                        async with self._lock:
                            self._memory_cache[h] = entry
                        return entry["data"]
            except Exception as e:
                logger.error(f"Error reading disk cache: {e}")
        
        return None

    async def set(self, key, data, category="default"):
        h = self._get_hash(f"{category}:{key}")
        entry = {"data": data, "ts": time.time()}
        
        # Update memory
        async with self._lock:
            self._memory_cache[h] = entry
            
        # Update disk
        category_dir = os.path.join(self.cache_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            
        path = os.path.join(category_dir, f"{h}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error writing disk cache: {e}")

# Global instance
cache_manager = CacheManager()
