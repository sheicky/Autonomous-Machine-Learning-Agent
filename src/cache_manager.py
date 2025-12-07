import os
import pickle
import hashlib
import json
import pandas as pd
from pathlib import Path

class CacheManager:
    """Intelligent caching for preprocessing and model artifacts."""
    
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_key(self, data, params=None):
        """Generate cache key from data and parameters."""
        if isinstance(data, pd.DataFrame):
            # Hash based on shape, columns, and sample of data
            key_str = f"{data.shape}_{list(data.columns)}_{data.head().to_json()}"
        else:
            key_str = str(data)
        
        if params:
            key_str += json.dumps(params, sort_keys=True)
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key, operation_type="preprocessing"):
        """Retrieve cached result."""
        cache_path = self.cache_dir / f"{operation_type}_{key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                
                # Update access time
                if key in self.metadata:
                    self.metadata[key]["last_accessed"] = pd.Timestamp.now().isoformat()
                    self.metadata[key]["access_count"] = self.metadata[key].get("access_count", 0) + 1
                    self._save_metadata()
                
                return result
            except Exception as e:
                print(f"Cache read error: {e}")
                return None
        
        return None
    
    def set(self, key, value, operation_type="preprocessing", metadata=None):
        """Store result in cache."""
        cache_path = self.cache_dir / f"{operation_type}_{key}.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Store metadata
            self.metadata[key] = {
                "operation_type": operation_type,
                "created": pd.Timestamp.now().isoformat(),
                "last_accessed": pd.Timestamp.now().isoformat(),
                "access_count": 0,
                "size_bytes": cache_path.stat().st_size,
                "metadata": metadata or {}
            }
            self._save_metadata()
            
            return True
        except Exception as e:
            print(f"Cache write error: {e}")
            return False
    
    def has(self, key, operation_type="preprocessing"):
        """Check if key exists in cache."""
        cache_path = self.cache_dir / f"{operation_type}_{key}.pkl"
        return cache_path.exists()
    
    def invalidate(self, key=None, operation_type=None):
        """Invalidate cache entries."""
        if key:
            # Invalidate specific key
            cache_path = self.cache_dir / f"{operation_type}_{key}.pkl"
            if cache_path.exists():
                cache_path.unlink()
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
        else:
            # Invalidate all for operation type
            pattern = f"{operation_type}_*" if operation_type else "*"
            for cache_file in self.cache_dir.glob(f"{pattern}.pkl"):
                cache_file.unlink()
            
            # Clear metadata
            if operation_type:
                self.metadata = {k: v for k, v in self.metadata.items() 
                               if v.get("operation_type") != operation_type}
            else:
                self.metadata = {}
            self._save_metadata()
    
    def get_stats(self):
        """Get cache statistics."""
        stats = {
            "total_entries": len(self.metadata),
            "total_size_mb": sum(v.get("size_bytes", 0) for v in self.metadata.values()) / (1024 * 1024),
            "by_operation": {}
        }
        
        for key, meta in self.metadata.items():
            op_type = meta.get("operation_type", "unknown")
            if op_type not in stats["by_operation"]:
                stats["by_operation"][op_type] = {"count": 0, "size_mb": 0}
            
            stats["by_operation"][op_type]["count"] += 1
            stats["by_operation"][op_type]["size_mb"] += meta.get("size_bytes", 0) / (1024 * 1024)
        
        return stats
    
    def cleanup_old(self, days=7):
        """Remove cache entries older than specified days."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        
        to_remove = []
        for key, meta in self.metadata.items():
            last_accessed = pd.Timestamp(meta.get("last_accessed", meta.get("created")))
            if last_accessed < cutoff:
                to_remove.append((key, meta.get("operation_type")))
        
        for key, op_type in to_remove:
            self.invalidate(key, op_type)
        
        return len(to_remove)
