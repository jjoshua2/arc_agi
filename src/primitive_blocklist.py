"""
Primitive blocklist manager for tracking problematic primitives.
Maintains a persistent list of primitives that consistently crash workers.
"""
import os
import json
import time
from typing import Dict, Set, List
from pathlib import Path
from collections import defaultdict, Counter

class PrimitiveBlocklist:
    """Manages a persistent blocklist of problematic primitives"""
    
    def __init__(self, blocklist_path: str = "primitive_blocklist.json"):
        self.blocklist_path = Path(blocklist_path)
        self.blocked_primitives: Set[str] = set()
        self.failure_counts: Counter = Counter()
        self.failure_history: Dict[str, List[float]] = defaultdict(list)  # primitive_id -> [timestamps]
        self.crash_threshold = 3  # Block after N crashes
        self.time_window = 3600.0  # Within 1 hour window
        self.load_blocklist()
    
    def load_blocklist(self):
        """Load blocklist from disk"""
        if self.blocklist_path.exists():
            try:
                with open(self.blocklist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.blocked_primitives = set(data.get('blocked_primitives', []))
                self.failure_counts = Counter(data.get('failure_counts', {}))
                
                # Convert timestamp lists back to floats
                history = data.get('failure_history', {})
                self.failure_history = {pid: list(timestamps) for pid, timestamps in history.items()}
                
                if self.blocked_primitives:
                    print(f"Loaded primitive blocklist: {len(self.blocked_primitives)} blocked primitives")
                    
            except Exception as e:
                print(f"Warning: failed to load primitive blocklist: {e}")
                self._reset_blocklist()
        else:
            self._reset_blocklist()
    
    def save_blocklist(self):
        """Save blocklist to disk"""
        try:
            # Clean old timestamps outside window
            current_time = time.time()
            cleaned_history = {}
            for pid, timestamps in self.failure_history.items():
                recent_timestamps = [ts for ts in timestamps if current_time - ts < self.time_window * 24]  # Keep 24h of history
                if recent_timestamps:
                    cleaned_history[pid] = recent_timestamps
            
            data = {
                'blocked_primitives': list(self.blocked_primitives),
                'failure_counts': dict(self.failure_counts),
                'failure_history': cleaned_history,
                'metadata': {
                    'crash_threshold': self.crash_threshold,
                    'time_window': self.time_window,
                    'last_updated': current_time
                }
            }
            
            with open(self.blocklist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: failed to save primitive blocklist: {e}")
    
    def _reset_blocklist(self):
        """Reset blocklist to empty state"""
        self.blocked_primitives = set()
        self.failure_counts = Counter()
        self.failure_history = defaultdict(list)
    
    def record_failure(self, primitive_id: str, error_type: str = "crash") -> bool:
        """
        Record a primitive failure. Returns True if primitive should now be blocked.
        
        Args:
            primitive_id: ID of the failing primitive
            error_type: Type of error ("crash", "timeout", "memory", etc.)
            
        Returns:
            True if primitive was newly added to blocklist
        """
        if primitive_id in self.blocked_primitives:
            return False  # Already blocked
        
        current_time = time.time()
        
        # Add to failure history
        self.failure_history[primitive_id].append(current_time)
        self.failure_counts[primitive_id] += 1
        
        # Count recent failures within time window
        recent_failures = [
            ts for ts in self.failure_history[primitive_id] 
            if current_time - ts < self.time_window
        ]
        
        # Check if should be blocked
        if len(recent_failures) >= self.crash_threshold:
            self.blocked_primitives.add(primitive_id)
            print(f"🚫 BLOCKED primitive {primitive_id} after {len(recent_failures)} failures in {self.time_window/60:.1f}m ({error_type})")
            self.save_blocklist()
            return True
        else:
            print(f"⚠️  Primitive {primitive_id} failed ({len(recent_failures)}/{self.crash_threshold} to block, {error_type})")
            if len(recent_failures) % 5 == 0:  # Save every 5 failures
                self.save_blocklist()
            return False
    
    def is_blocked(self, primitive_id: str) -> bool:
        """Check if primitive is blocked"""
        return primitive_id in self.blocked_primitives
    
    def filter_primitives(self, primitives: List, get_id_func=None) -> List:
        """
        Filter out blocked primitives from a list
        
        Args:
            primitives: List of primitive objects
            get_id_func: Function to extract ID from primitive (default: getattr(p, 'id'))
            
        Returns:
            Filtered list with blocked primitives removed
        """
        if not self.blocked_primitives:
            return primitives
        
        if get_id_func is None:
            get_id_func = lambda p: getattr(p, 'id', 'unknown')
        
        filtered = []
        blocked_count = 0
        
        for primitive in primitives:
            primitive_id = get_id_func(primitive)
            if self.is_blocked(primitive_id):
                blocked_count += 1
            else:
                filtered.append(primitive)
        
        if blocked_count > 0:
            print(f"🚫 Filtered out {blocked_count} blocked primitives ({len(filtered)}/{len(primitives)} remaining)")
        
        return filtered
    
    def get_statistics(self) -> Dict:
        """Get blocklist statistics"""
        current_time = time.time()
        
        # Count recent failures across all primitives
        recent_failure_count = 0
        for timestamps in self.failure_history.values():
            recent_failure_count += len([ts for ts in timestamps if current_time - ts < self.time_window])
        
        return {
            'total_blocked': len(self.blocked_primitives),
            'total_failures_recorded': sum(self.failure_counts.values()),
            'recent_failures': recent_failure_count,
            'most_problematic': self.failure_counts.most_common(10),
            'blocked_primitives': list(self.blocked_primitives)
        }
    
    def unblock_primitive(self, primitive_id: str) -> bool:
        """
        Manually unblock a primitive (for testing/debugging)
        
        Returns:
            True if primitive was unblocked, False if it wasn't blocked
        """
        if primitive_id in self.blocked_primitives:
            self.blocked_primitives.remove(primitive_id)
            print(f"✅ Unblocked primitive {primitive_id}")
            self.save_blocklist()
            return True
        return False
    
    def clear_blocklist(self):
        """Clear all blocked primitives (nuclear option)"""
        count = len(self.blocked_primitives)
        self._reset_blocklist()
        self.save_blocklist()
        print(f"🗑️  Cleared blocklist ({count} primitives unblocked)")

# Global instance
_global_blocklist: PrimitiveBlocklist = None

def get_primitive_blocklist() -> PrimitiveBlocklist:
    """Get or create global primitive blocklist"""
    global _global_blocklist
    if _global_blocklist is None:
        _global_blocklist = PrimitiveBlocklist()
    return _global_blocklist