"""
Fast persistent worker pool for primitive evaluation.
Replaces per-call process spawning with reusable workers.
"""
import math
import os
import json
import time
import traceback
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

from src.models import GRID
from src.run_python import run_python_transform_sync

try:
    import signal as _signal
    _HAS_SETITIMER = hasattr(_signal, "setitimer")
except Exception:
    _signal = None
    _HAS_SETITIMER = False

def _timeout_handler(signum, frame):
    raise TimeoutError("primitive execution timed out")

_INLINE_TIMEOUT_DEFAULT = float(os.environ.get("ARC_FAST_SWEEP_PRIMITIVE_TIMEOUT", "2"))

# Global worker state (populated by initializer)
_PRIMITIVES_CACHE: Dict[str, str] = {}  # id -> python_code_str
_WORKER_ID: Optional[int] = None
# Per-batch cached grids in each worker
_BATCH_INPUTS: Optional[List[GRID]] = None
_BATCH_OUTPUTS: Optional[List[GRID]] = None
_BATCH_HASH: Optional[str] = None

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
        # Reset batch context
        global _BATCH_INPUTS, _BATCH_OUTPUTS, _BATCH_HASH
        _BATCH_INPUTS, _BATCH_OUTPUTS, _BATCH_HASH = None, None, None
    except Exception as e:
        print(f"❌ Worker {_WORKER_ID} init failed: {e}")
        import traceback
        print(f"Worker {_WORKER_ID} traceback: {traceback.format_exc()}")

def _worker_set_batch_context(train_inputs: List[GRID], train_outputs: List[GRID], ctx_hash: str) -> int:
    """Set per-batch grids in the worker and return worker id"""
    global _BATCH_INPUTS, _BATCH_OUTPUTS, _BATCH_HASH, _WORKER_ID
    _BATCH_INPUTS = train_inputs
    _BATCH_OUTPUTS = train_outputs
    _BATCH_HASH = ctx_hash
    return _WORKER_ID or -1


def _evaluate_primitive_in_worker(job: EvalJob) -> PrimitiveResult:
    """Evaluate single primitive in worker process"""
    global _PRIMITIVES_CACHE
    start_time = time.perf_counter()
    print(f"▶️  Worker {_WORKER_ID}: starting primitive {job.primitive_id}")
    
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
        
        # Resolve grids from job or batch cache
        grids_in = job.train_inputs if job.train_inputs is not None else _BATCH_INPUTS
        grids_out = job.train_outputs if job.train_outputs is not None else _BATCH_OUTPUTS
        if grids_in is None or grids_out is None:
            raise RuntimeError("Worker batch context not set and no grids provided in job")
        timeout_sec = float(job.timeout_sec or _INLINE_TIMEOUT_DEFAULT)
        failure_reason = "Transform failed"

        timed_out = False
        if _HAS_SETITIMER:
            results, timed_out = _run_transform_inline(code, grids_in, timeout_sec=timeout_sec)
            if timed_out:
                failure_reason = f"Timeout after {timeout_sec:.2f}s"
        else:
            timeout_int = max(1, int(math.ceil(timeout_sec)))
            run_result = run_python_transform_sync(
                code=code,
                grid_lists=grids_in,
                timeout=timeout_int,
                raise_exception=False,
            )
            if run_result and run_result.transform_results:
                results = run_result.transform_results
                timed_out = False
            else:
                results = None
                timed_out = bool(run_result and run_result.timed_out)
                if run_result:
                    if run_result.timed_out:
                        failure_reason = f"Timeout after {timeout_int}s"
                    elif run_result.stderr:
                        failure_reason = run_result.stderr.strip()[:200]
                else:
                    failure_reason = "Transform failed"

        if not results:
            if timed_out:
                print(f"⚠️  Worker {_WORKER_ID}: primitive {job.primitive_id} timed out after {timeout_sec:.2f}s")
            return PrimitiveResult(
                primitive_id=job.primitive_id,
                num_correct=0.0,
                accuracy_score=0.0,
                success=False,
                error_msg=failure_reason,
            )

        # Compute scores
        num_correct = 0.0
        accuracy_scores = []
        
        for i, (expected, actual) in enumerate(zip(grids_out, results)):
            if expected == actual:
                num_correct += 1.0
            acc = _percent_correct_cells(expected, actual)
            accuracy_scores.append(acc)
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        eval_time_ms = (time.perf_counter() - start_time) * 1000
        
        result = PrimitiveResult(
            primitive_id=job.primitive_id,
            num_correct=num_correct,
            accuracy_score=avg_accuracy,
            success=True,
            eval_time_ms=eval_time_ms
        )
        print(f"✅ Worker {_WORKER_ID}: finished primitive {job.primitive_id} in {eval_time_ms:.2f}ms")
        return result
        
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
        print(f"⚠️  Worker {_WORKER_ID}: exception during primitive {job.primitive_id}: {e}")
        return PrimitiveResult(
            primitive_id=job.primitive_id,
            num_correct=0.0,
            accuracy_score=0.0,
            success=False,
            error_msg=str(e)[:200],  # Truncate long errors
            eval_time_ms=eval_time_ms
        )


def _evaluate_primitives_chunk_in_worker(primitive_ids, train_inputs, train_outputs, timeout_sec, ctx_hash: Optional[str] = None):
    """Evaluate a chunk of primitive IDs inside a single worker call"""
    # Make chunk self-contained: always use provided grids, avoid global batch context
    results: list[PrimitiveResult] = []
    for pid in primitive_ids:
        job = EvalJob(primitive_id=pid, train_inputs=train_inputs, train_outputs=train_outputs, timeout_sec=timeout_sec)
        res = _evaluate_primitive_in_worker(job)
        results.append(res)
    return results

def _run_transform_inline(code: str, grid_inputs: List[GRID], timeout_sec: float) -> tuple[Optional[List[GRID]], bool]:
    """Run transform function inline with optional POSIX timer fallback.

    Returns: (results, timed_out)
    """
    try:
        # Defensive programming: limit code length to prevent memory issues
        if len(code) > 50000:  # 50KB limit
            return None, False
            
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
        
        # Execute the code to define transform function with timeout if supported
        prev_handler = None
        try:
            if _HAS_SETITIMER:
                prev_handler = _signal.getsignal(_signal.SIGALRM)
                _signal.signal(_signal.SIGALRM, _timeout_handler)
                _signal.setitimer(_signal.ITIMER_REAL, max(0.1, timeout_sec))
            exec(code, exec_globals)
        except TimeoutError:
            return None, True
        except (MemoryError, RecursionError, SystemError):
            return None, False
        except Exception:
            return None, False
        finally:
            if _HAS_SETITIMER:
                _signal.setitimer(_signal.ITIMER_REAL, 0.0)
                if prev_handler is not None:
                    _signal.signal(_signal.SIGALRM, prev_handler)
            
        transform_func = exec_globals.get('transform')
        
        if not callable(transform_func):
            return None, False

        # Apply to all inputs with individual timeout/error handling
        results = []
        for grid_input in grid_inputs:
            try:
                # Basic input validation
                if not _is_valid_grid(grid_input):
                    return None, False
                if _HAS_SETITIMER:
                    _signal.setitimer(_signal.ITIMER_REAL, max(0.1, timeout_sec))
                result = transform_func(grid_input)
                
                # Validate result
                if not _is_valid_grid(result):
                    return None, False
                    
                # Size sanity check (prevent memory bombs)
                if len(result) > 100 or (result and len(result[0]) > 100):
                    return None, False
                    
                results.append(result)
            except TimeoutError:
                if _HAS_SETITIMER:
                    _signal.setitimer(_signal.ITIMER_REAL, 0.0)
                return None, True
            except (MemoryError, RecursionError, SystemError):
                # Worker-killing errors
                return None, False
            except Exception:
                # Regular transform errors
                return None, False
            finally:
                if _HAS_SETITIMER:
                    _signal.setitimer(_signal.ITIMER_REAL, 0.0)
                
        return results, False
        
    except (MemoryError, RecursionError, SystemError):
        # Top-level worker protection
        return None, False
    except Exception:
        return None, False

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
        self._last_ctx_hash: Optional[str] = None
        
    def start(self):
        """Start the worker pool"""
        if self._executor is not None:
            return  # Already started
            
        # Use spawn context for robustness across notebooks/process restarts
        ctx = mp.get_context('spawn')
        
        self._executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_worker_initializer,
            initargs=(self._library_data,),
            max_tasks_per_child=self.max_tasks_per_child
        )
        
        print(f"🚀 Started transform pool: {self.num_workers} workers, spawn context, {self.max_tasks_per_child} tasks per child (persistent across challenges)")
        
    def shutdown(self):
        """Shutdown the worker pool"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            
    def _compute_ctx_hash(self, train_inputs: List[GRID], train_outputs: List[GRID]) -> str:
        import hashlib, json
        def mini(g):
            # shapes + first row checksum
            if not g:
                return [0,0,0]
            r, c = len(g), len(g[0]) if g[0] else 0
            s = sum(g[0]) if g and g[0] else 0
            return [r,c,int(s)%997]
        payload = {
            'in': [mini(x) for x in train_inputs],
            'out': [mini(x) for x in train_outputs],
        }
        return hashlib.md5(json.dumps(payload, separators=(',',':')).encode()).hexdigest()[:12]

    def _ensure_batch_context(self, train_inputs: List[GRID], train_outputs: List[GRID], ctx_hash: str):
        # Broadcast grids to all workers (best-effort, ensure each worker sets context)
        seen: set[int] = set()
        attempts = 0
        futures = []
        while len(seen) < self.num_workers and attempts < self.num_workers * 3:
            fut = self._executor.submit(_worker_set_batch_context, train_inputs, train_outputs, ctx_hash)
            futures.append(fut)
            attempts += 1
        # Collect and record worker ids
        for fut in futures:
            try:
                wid = fut.result(timeout=30)
                if isinstance(wid, int) and wid != -1:
                    seen.add(wid)
            except Exception:
                pass
        print(f"Broadcasted batch context to {len(seen)}/{self.num_workers} workers (ctx={ctx_hash})")

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
            
        # Compute context hash for per-challenge state
        ctx_hash = self._compute_ctx_hash(train_inputs, train_outputs)
        # Optionally restart pool per challenge to avoid lingering state
        restart_per_challenge = os.environ.get("ARC_FAST_SWEEP_RESTART_PER_CHALLENGE", "0") == "1"
        if restart_per_challenge and (self._last_ctx_hash is not None) and (self._last_ctx_hash != ctx_hash):
            print("🔄 New challenge detected; restarting transform pool to reset worker state")
            try:
                self.shutdown()
            except Exception:
                pass
            self.start()
        self._last_ctx_hash = ctx_hash
        
        # Optionally broadcast batch context (disabled by default)
        if os.environ.get("ARC_FAST_SWEEP_BROADCAST", "0") == "1":
            self._ensure_batch_context(train_inputs, train_outputs, ctx_hash)

        # Chunk primitives to reduce IPC/pickling overhead
        try:
            chunk_size = max(16, int(os.environ.get("ARC_FAST_SWEEP_CHUNK", "64")))
        except Exception:
            chunk_size = 64
        chunks: List[List[str]] = [primitive_ids[i:i+chunk_size] for i in range(0, len(primitive_ids), chunk_size)]
        
        # Stream chunk jobs to avoid synchronous submit overhead
        chunk_queue = deque(chunks)
        future_to_chunk: Dict[Any, List[str]] = {}
        inflight_limit = min(self.num_workers, len(chunks))
        for _ in range(inflight_limit):
            if not chunk_queue:
                break
            ch = chunk_queue.popleft()
            fut = self._executor.submit(
                _evaluate_primitives_chunk_in_worker,
                ch, train_inputs, train_outputs, timeout_per_primitive, ctx_hash
            )
            future_to_chunk[fut] = ch
        print(f"Streaming {len(chunks)} chunk-jobs (chunk_size={chunk_size}, inflight={inflight_limit})")
        
        # Collect results with progress tracking and timeout + stall watchdog
        results: List[PrimitiveResult] = []
        completed_count = 0
        total_jobs = len(primitive_ids)
        
        # Add overall timeout to prevent infinite hangs
        import concurrent.futures
        # Tighter cap: default fast sweep should finish well under a minute; bail out early on hangs
        timeout_seconds = max(120, min(300, int(total_jobs * 0.2)))  # 0.2s per primitive, min 120s, max 300s
        print(f"Starting evaluation of {total_jobs} primitives (timeout: {timeout_seconds}s)")
        
        # Stall watchdog: if no completions for N seconds, restart pool and re-submit remaining
        try:
            stall_env = os.environ.get("ARC_FAST_SWEEP_STALL_SECS")
            stall_secs = int(stall_env) if stall_env else max(30, min(120, int(chunk_size * 0.4)))
        except Exception:
            stall_secs = max(30, min(120, int(chunk_size * 0.4)))
        last_progress = time.perf_counter()
        
        try:
            while completed_count < total_jobs:
                try:
                    for future in as_completed(list(future_to_chunk.keys()), timeout=5):
                        try:
                            chunk_results = future.result()
                            # Aggregate
                            results.extend(chunk_results)
                            completed_count += len(chunk_results)
                            last_progress = time.perf_counter()
                            
                            # Progress logging at 50% and 100% only
                            progress_pct = completed_count / total_jobs * 100
                            halfway = total_jobs // 2
                            if completed_count == halfway or completed_count == total_jobs:
                                print(f"Progress: {completed_count}/{total_jobs} primitives evaluated ({progress_pct:.1f}%)")
                            
                            # Blocklist worker-killing errors
                            for r in chunk_results:
                                if not r.success and r.error_msg and r.error_msg.startswith("WORKER_KILLER:"):
                                    from src.primitive_blocklist import get_primitive_blocklist
                                    blocklist = get_primitive_blocklist()
                                    error_parts = r.error_msg.split(":", 2)
                                    error_type = error_parts[1] if len(error_parts) > 1 else "crash"
                                    blocklist.record_failure(r.primitive_id, error_type.lower())
                                    print(f"⚠️  Blocked primitive {r.primitive_id} due to {error_type}")
                            
                            # Remove processed future
                            future_to_chunk.pop(future, None)
                            # Submit next chunk if available
                            if chunk_queue:
                                ch_next = chunk_queue.popleft()
                                fut_next = self._executor.submit(
                                    _evaluate_primitives_chunk_in_worker,
                                    ch_next, train_inputs, train_outputs, timeout_per_primitive, ctx_hash
                                )
                                future_to_chunk[fut_next] = ch_next
                        except Exception as e:
                            chunk = future_to_chunk.get(future, [])
                            # Check if this is a worker crash that killed the pool
                            if "process pool" in str(e).lower() or "terminated abruptly" in str(e).lower():
                                chunk_str = ",".join(chunk) if chunk else "<unknown>"
                                print(f"Worker pool crashed while processing chunk [{chunk_str}]: {e}")
                                try:
                                    import traceback
                                    trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                                    print(trace.rstrip())
                                except Exception:
                                    pass
                                print(f"Worker pool crashed, attempting to recreate...")
                                try:
                                    self.shutdown()
                                    self.start()
                                    print("Worker pool recreated successfully")
                                    # Don't return partial results, let caller retry
                                    raise RuntimeError("Worker pool was recreated, please retry") from e
                                except Exception as restart_e:
                                    print(f"Failed to recreate worker pool: {restart_e}")
                                    raise e  # Original error

                            # Record crashes for all pids in chunk
                            for pid in chunk:
                                print(f"🚨 Worker crashed during primitive {pid}: {e}")
                                results.append(PrimitiveResult(
                                    primitive_id=pid,
                                    num_correct=0.0,
                                    accuracy_score=0.0,
                                    success=False,
                                    error_msg=f"Worker exception: {str(e)[:100]}"
                                ))
                            future_to_chunk.pop(future, None)
                except concurrent.futures.TimeoutError:
                    # No completions in the last short window; check for stall
                    if (time.perf_counter() - last_progress) > stall_secs and future_to_chunk:
                        print(f"⏱️  Stall detected (>{stall_secs}s without progress). Restarting pool and resubmitting remaining chunks...")
                        remaining_chunks = [chunk for fut, chunk in list(future_to_chunk.items()) if not fut.done()]
                        # Include queued chunks
                        remaining_chunks.extend(list(chunk_queue))
                        # Restart pool
                        try:
                            self.shutdown()
                        except Exception:
                            pass
                        self.start()
                        # Re-prime inflight
                        future_to_chunk = {}
                        chunk_queue = deque(remaining_chunks)
                        for _ in range(inflight_limit):
                            if not chunk_queue:
                                break
                            ch = chunk_queue.popleft()
                            fut = self._executor.submit(
                                _evaluate_primitives_chunk_in_worker,
                                ch, train_inputs, train_outputs, timeout_per_primitive, ctx_hash
                            )
                            future_to_chunk[fut] = ch
                        last_progress = time.perf_counter()
                        continue
                
                if not future_to_chunk:
                    break
        
        except concurrent.futures.TimeoutError:
            print(f"❌ TIMEOUT: Only completed {completed_count}/{total_jobs} primitives in {timeout_seconds}s")
            print("This suggests workers are hanging or dying. Cancelling remaining futures...")
            
            # Cancel remaining futures and collect what we have
            for future, chunk in future_to_chunk.items():
                if not future.done():
                    future.cancel()
                    for pid in chunk:
                        results.append(PrimitiveResult(
                            primitive_id=pid,
                            num_correct=0.0,
                            accuracy_score=0.0,
                            success=False,
                            error_msg="Evaluation timeout - worker may have hung"
                        ))
            
            print(f"Returning {len(results)} results (some may be timeouts)")
        
        return results

# Global pool instance (simple startup initialization)
_global_pool: Optional[FastTransformPool] = None

def initialize_global_pool(library) -> None:
    """Initialize the global transform pool at startup"""
    global _global_pool
    
    if _global_pool is not None:
        print("⚠️  Global pool already initialized, skipping")
        return
        
    print(f"🏗️  Initializing global transform pool at startup ({len(library.primitives)} primitives)")
    _global_pool = FastTransformPool(library)
    _global_pool.start()
    print(f"✅ Global transform pool ready")

def get_global_transform_pool(library=None) -> FastTransformPool:
    """Get the global transform pool, initializing on-demand if needed"""
    global _global_pool
    if _global_pool is None:
        if library is None:
            raise RuntimeError("Global pool not initialized and no library provided for on-demand init")
        print("🏗️  Initializing global transform pool on-demand")
        _global_pool = FastTransformPool(library)
        _global_pool.start()
        print("✅ Global transform pool ready")
    if not _global_pool._executor:
        print("🔄 Restarting shutdown pool...")
        _global_pool.start()
    return _global_pool

def shutdown_global_pool():
    """Shutdown global pool"""
    global _global_pool
    if _global_pool:
        print("🔄 Shutting down global transform pool...")
        _global_pool.shutdown()
        _global_pool = None
        print("✅ Global transform pool shutdown complete")
