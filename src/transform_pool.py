"""
Fast persistent worker pool for primitive evaluation.
Replaces per-call process spawning with reusable workers.
"""
import os
import json
import time
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np

from src.models import GRID

# Global worker state (populated by initializer)
_PRIMITIVES_CACHE: Dict[str, str] = {}  # id -> python_code_str
_WORKER_ID: Optional[int] = None

@dataclass
class PrimitiveResult:
    """Compact result from worker evaluation"""
    primitive_id: str
    num_correct: float
    accuracy_score: float
    success: bool
    error_msg: Optional[str] = None
    eval_time_ms: float = 0.0

@dataclass
class EvalJob:
    """Job spec sent to worker"""
    primitive_id: str
    train_inputs: List[GRID]
    train_outputs: List[GRID] 
    timeout_sec: float = 5.0

def _worker_initializer(library_data: bytes):
    """Initialize worker process with library cache"""
    global _PRIMITIVES_CACHE, _WORKER_ID
    try:
        # Get actual process ID for logging
        import os
        _WORKER_ID = os.getpid()
        # Only log first few workers to reduce noise
        if _WORKER_ID % 1000 == _WORKER_ID or _WORKER_ID < 1005:
            print(f"🔧 Worker {_WORKER_ID}: Starting initialization...")
        
        # Deserialize library
        import pickle
        library = pickle.loads(library_data)
        _PRIMITIVES_CACHE = {
            getattr(p, 'id', f'prim_{i}'): p.python_code_str 
            for i, p in enumerate(library.primitives)
        }
        # Set numpy to be quiet
        np.seterr(all='ignore')
        
        # Only log first few workers to reduce noise
        if _WORKER_ID % 1000 == _WORKER_ID or _WORKER_ID < 1005:
            print(f"✅ Worker {_WORKER_ID}: loaded {len(_PRIMITIVES_CACHE)} primitives")
    except Exception as e:
        print(f"❌ Worker {_WORKER_ID} init failed: {e}")
        import traceback
        print(f"Worker {_WORKER_ID} traceback: {traceback.format_exc()}")

def _evaluate_primitive_in_worker(job: EvalJob) -> PrimitiveResult:
    """Evaluate single primitive in worker process"""
    global _PRIMITIVES_CACHE
    
    start_time = time.perf_counter()
    
    try:
        code = _PRIMITIVES_CACHE.get(job.primitive_id)
        if not code:
            return PrimitiveResult(
                primitive_id=job.primitive_id,
                num_correct=0.0,
                accuracy_score=0.0,
                success=False,
                error_msg="Primitive not found in cache"
            )
        
        # Execute transform inline (avoid subprocess)
        results = _run_transform_inline(code, job.train_inputs, job.timeout_sec)
        
        if not results:
            return PrimitiveResult(
                primitive_id=job.primitive_id,
                num_correct=0.0,
                accuracy_score=0.0,
                success=False,
                error_msg="Transform failed"
            )
        
        # Compute scores
        num_correct = 0.0
        accuracy_scores = []
        
        for i, (expected, actual) in enumerate(zip(job.train_outputs, results)):
            if expected == actual:
                num_correct += 1.0
            acc = _percent_correct_cells(expected, actual)
            accuracy_scores.append(acc)
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        eval_time_ms = (time.perf_counter() - start_time) * 1000
        
        return PrimitiveResult(
            primitive_id=job.primitive_id,
            num_correct=num_correct,
            accuracy_score=avg_accuracy,
            success=True,
            eval_time_ms=eval_time_ms
        )
        
    except (MemoryError, RecursionError, SystemError) as e:
        # These are worker-killing errors - record them for blocklist
        eval_time_ms = (time.perf_counter() - start_time) * 1000
        error_type = type(e).__name__
        print(f"🚨 Worker {_WORKER_ID}: KILLER ERROR on {job.primitive_id}: {error_type}")
        return PrimitiveResult(
            primitive_id=job.primitive_id,
            num_correct=0.0,
            accuracy_score=0.0,
            success=False,
            error_msg=f"WORKER_KILLER:{error_type}:{str(e)[:150]}",
            eval_time_ms=eval_time_ms
        )
    except Exception as e:
        eval_time_ms = (time.perf_counter() - start_time) * 1000
        return PrimitiveResult(
            primitive_id=job.primitive_id,
            num_correct=0.0,
            accuracy_score=0.0,
            success=False,
            error_msg=str(e)[:200],  # Truncate long errors
            eval_time_ms=eval_time_ms
        )

def _run_transform_inline(code: str, grid_inputs: List[GRID], timeout_sec: float) -> Optional[List[GRID]]:
    """Run transform function inline without subprocess"""
    try:
        # Defensive programming: limit code length to prevent memory issues
        if len(code) > 50000:  # 50KB limit
            return None
            
        # Create safe execution environment with limited builtins
        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'list': list, 'dict': dict, 'set': set,
            'min': min, 'max': max, 'sum': sum, 'abs': abs,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'True': True, 'False': False, 'None': None,
        }
        
        exec_globals = {
            '__builtins__': safe_builtins,
            'np': np,
            'numpy': np,
            'json': json,
            'List': List,
            'Tuple': Tuple,
            'Set': set,
            'Union': Any,
            'Optional': Optional,
        }
        
        # Execute the code to define transform function with timeout simulation
        try:
            exec(code, exec_globals)
        except (MemoryError, RecursionError, SystemError) as e:
            # These can crash the worker, return None instead
            return None
        except Exception:
            # Other execution errors
            return None
            
        transform_func = exec_globals.get('transform')
        
        if not callable(transform_func):
            return None
        
        # Apply to all inputs with individual timeout/error handling
        results = []
        for grid_input in grid_inputs:
            try:
                # Basic input validation
                if not _is_valid_grid(grid_input):
                    return None
                    
                result = transform_func(grid_input)
                
                # Validate result
                if not _is_valid_grid(result):
                    return None
                    
                # Size sanity check (prevent memory bombs)
                if len(result) > 100 or (result and len(result[0]) > 100):
                    return None
                    
                results.append(result)
            except (MemoryError, RecursionError, SystemError):
                # Worker-killing errors
                return None
            except Exception:
                # Regular transform errors
                return None
                
        return results
        
    except (MemoryError, RecursionError, SystemError):
        # Top-level worker protection
        return None
    except Exception:
        return None

def _is_valid_grid(obj) -> bool:
    """Fast grid validation"""
    try:
        if not isinstance(obj, list):
            return False
        if not obj:  # Empty grid
            return True
        if not all(isinstance(row, list) for row in obj):
            return False
        if not all(isinstance(cell, int) for row in obj for cell in row):
            return False
        return True
    except:
        return False

def _percent_correct_cells(expected: GRID, actual: GRID) -> float:
    """Compute percentage of correct cells"""
    try:
        if len(expected) != len(actual):
            return 0.0
        if not expected:
            return 1.0 if not actual else 0.0
        if len(expected[0]) != len(actual[0]):
            return 0.0
        
        total_cells = len(expected) * len(expected[0])
        if total_cells == 0:
            return 1.0
            
        correct = sum(
            1 for i in range(len(expected))
            for j in range(len(expected[0]))
            if expected[i][j] == actual[i][j]
        )
        return correct / total_cells
        
    except:
        return 0.0

class FastTransformPool:
    """Persistent worker pool for primitive evaluation"""
    
    def __init__(self, 
                 library,
                 num_workers: int = None,
                 max_tasks_per_child: int = 200):
        """
        Args:
            library: Library object with primitives
            num_workers: Number of worker processes (default: CPU count)  
            max_tasks_per_child: Restart workers after N tasks
        """
        if num_workers is None:
            # Configurable workers with memory-conscious defaults
            try:
                max_workers = int(os.environ.get("ARC_FAST_SWEEP_WORKERS", "4"))
                num_workers = min(max_workers, os.cpu_count() or 4)
            except:
                num_workers = 2  # Very conservative default for memory
            
        self.num_workers = num_workers
        # Set high task limit so workers persist through entire challenges
        # Each challenge ~2000+ primitives, allow multiple challenges per worker
        self.max_tasks_per_child = max(max_tasks_per_child, 10000)
        
        # Serialize library for workers (this might be large!)
        import pickle
        self._library_data = pickle.dumps(library)
        print(f"Serialized library: {len(self._library_data) / 1024 / 1024:.1f}MB")
        self._executor = None
        
    def start(self):
        """Start the worker pool"""
        if self._executor is not None:
            return  # Already started
            
        # Use forkserver on Linux for notebook compatibility
        ctx = mp.get_context('forkserver' if hasattr(mp, 'get_context') else 'spawn')
        
        self._executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_worker_initializer,
            initargs=(self._library_data,),
            max_tasks_per_child=self.max_tasks_per_child
        )
        
        print(f"🚀 Started transform pool: {self.num_workers} workers, forkserver context, {self.max_tasks_per_child} tasks per child (persistent across challenges)")
        
    def shutdown(self):
        """Shutdown the worker pool"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            
    def evaluate_primitives_batch(self, 
                                 primitive_ids: List[str],
                                 train_inputs: List[GRID], 
                                 train_outputs: List[GRID],
                                 timeout_per_primitive: float = 5.0) -> List[PrimitiveResult]:
        """
        Evaluate multiple primitives in parallel
        
        Args:
            primitive_ids: List of primitive IDs to evaluate
            train_inputs: Training input grids
            train_outputs: Expected training output grids  
            timeout_per_primitive: Timeout per primitive (not enforced strictly)
            
        Returns:
            List of PrimitiveResult objects
        """
        if not self._executor:
            self.start()
            
        # Create jobs
        jobs = [
            EvalJob(
                primitive_id=pid,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                timeout_sec=timeout_per_primitive
            )
            for pid in primitive_ids
        ]
        
        # Submit all jobs with timing
        submit_start = time.perf_counter()
        future_to_job = {
            self._executor.submit(_evaluate_primitive_in_worker, job): job 
            for job in jobs
        }
        submit_time = time.perf_counter() - submit_start
        print(f"Submitted {len(jobs)} jobs to pool in {submit_time:.3f}s")
        
        # Collect results with progress tracking and timeout
        results = []
        completed_count = 0
        total_jobs = len(jobs)
        
        # Add overall timeout to prevent infinite hangs (should be fast: ~2000 * 0.3s = 600s max)
        import concurrent.futures
        timeout_seconds = max(600, total_jobs * 0.5)  # 0.5s per primitive as safety margin
        print(f"Starting evaluation of {total_jobs} primitives (timeout: {timeout_seconds}s)")
        
        try:
            for future in as_completed(future_to_job, timeout=timeout_seconds):
                try:
                    result = future.result()  # Let primitives run (should be fast ~100-300ms)
                    completed_count += 1
                    
                    # Progress logging at 50% and 100% only
                    progress_pct = completed_count / total_jobs * 100
                    halfway = total_jobs // 2
                    if completed_count == halfway or completed_count == total_jobs:
                        print(f"Progress: {completed_count}/{total_jobs} primitives evaluated ({progress_pct:.1f}%)")
                    
                    # Check for worker-killing errors and record for blocklist
                    if not result.success and result.error_msg and result.error_msg.startswith("WORKER_KILLER:"):
                        from src.primitive_blocklist import get_primitive_blocklist
                        blocklist = get_primitive_blocklist()
                        error_parts = result.error_msg.split(":", 2)
                        error_type = error_parts[1] if len(error_parts) > 1 else "crash"
                        blocklist.record_failure(result.primitive_id, error_type.lower())
                        print(f"⚠️  Blocked primitive {result.primitive_id} due to {error_type}")
                    
                    results.append(result)
                    
                except Exception as e:
                    job = future_to_job[future]
                    # Check if this is a worker crash that killed the pool
                    if "process pool" in str(e).lower() or "terminated abruptly" in str(e).lower():
                        print(f"Worker pool crashed, attempting to recreate...")
                        try:
                            self.shutdown()
                            self.start()
                            print("Worker pool recreated successfully")
                            # Don't return partial results, let caller retry
                            raise Exception("Worker pool was recreated, please retry")
                        except Exception as restart_e:
                            print(f"Failed to recreate worker pool: {restart_e}")
                            raise e  # Original error
                    
                    # Record crashes for blocklist
                    print(f"🚨 Worker crashed during primitive {job.primitive_id}: {e}")
                    from src.primitive_blocklist import get_primitive_blocklist
                    blocklist = get_primitive_blocklist()
                    blocklist.record_failure(job.primitive_id, "worker_crash")
                    
                    results.append(PrimitiveResult(
                        primitive_id=job.primitive_id,
                        num_correct=0.0,
                        accuracy_score=0.0,
                        success=False,
                        error_msg=f"Worker exception: {str(e)[:100]}"
                    ))
        
        except concurrent.futures.TimeoutError:
            print(f"❌ TIMEOUT: Only completed {completed_count}/{total_jobs} primitives in {timeout_seconds}s")
            print("This suggests workers are hanging or dying. Cancelling remaining futures...")
            
            # Cancel remaining futures and collect what we have
            for future in future_to_job:
                if not future.done():
                    future.cancel()
                    job = future_to_job[future]
                    results.append(PrimitiveResult(
                        primitive_id=job.primitive_id,
                        num_correct=0.0,
                        accuracy_score=0.0,
                        success=False,
                        error_msg="Evaluation timeout - worker may have hung"
                    ))
            
            print(f"Returning {len(results)} results (some may be timeouts)")
        
        return results

import threading

# Global pool instance (singleton pattern for notebook compatibility)
_global_pool: Optional[FastTransformPool] = None
_pool_lock = threading.Lock()  # Thread-safe pool creation
_global_library_hash: Optional[str] = None  # Track library content hash

def _get_library_hash(library) -> str:
    """Get a hash of the library content for comparison"""
    try:
        import hashlib
        # Use primitive count and first/last primitive IDs as a simple hash
        primitive_ids = [getattr(p, 'id', f'prim_{i}') for i, p in enumerate(library.primitives)]
        content = f"{len(primitive_ids)}:{primitive_ids[0] if primitive_ids else ''}:{primitive_ids[-1] if primitive_ids else ''}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"

def get_global_transform_pool(library=None) -> FastTransformPool:
    """Get or create global transform pool (thread-safe, reuses same library)"""
    global _global_pool, _global_library_hash
    
    with _pool_lock:
        current_library_hash = _get_library_hash(library) if library else None
        
        if _global_pool is None:
            if library is not None:
                print(f"🏗️  Creating NEW global transform pool (first time)")
                _global_pool = FastTransformPool(library)
                _global_pool.start()
                _global_library_hash = current_library_hash
            else:
                print("⚠️  Cannot create pool: no library provided")
                return None
        elif not _global_pool._executor:
            # Pool exists but is shutdown, restart it
            print("🔄 Restarting existing transform pool...")
            _global_pool.start()
        elif current_library_hash and current_library_hash != _global_library_hash:
            # Library has changed, recreate pool
            print(f"🔄 Library changed (hash {_global_library_hash} -> {current_library_hash}), recreating pool...")
            _global_pool.shutdown()
            _global_pool = FastTransformPool(library)
            _global_pool.start()
            _global_library_hash = current_library_hash
        else:
            # Pool exists and library is the same - reuse silently
            pass
    
    return _global_pool

def shutdown_global_pool():
    """Shutdown global pool"""
    global _global_pool
    if _global_pool:
        _global_pool.shutdown()
        _global_pool = None