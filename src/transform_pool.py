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
        # Deserialize library
        import pickle
        library = pickle.loads(library_data)
        _PRIMITIVES_CACHE = {
            getattr(p, 'id', f'prim_{i}'): p.python_code_str 
            for i, p in enumerate(library.primitives)
        }
        # Set numpy to be quiet
        np.seterr(all='ignore')
        print(f"Worker {_WORKER_ID}: loaded {len(_PRIMITIVES_CACHE)} primitives")
    except Exception as e:
        print(f"Worker {_WORKER_ID} init failed: {e}")

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
            num_workers = min(4, os.cpu_count() or 4)  # Cap at 4 for Kaggle
            
        self.num_workers = num_workers
        self.max_tasks_per_child = max_tasks_per_child
        
        # Serialize library for workers
        self._library_data = pickle.dumps(library)
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
        
        print(f"Started transform pool: {self.num_workers} workers, forkserver context")
        
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
        
        # Submit all jobs
        future_to_job = {
            self._executor.submit(_evaluate_primitive_in_worker, job): job 
            for job in jobs
        }
        
        # Collect results
        results = []
        for future in as_completed(future_to_job):
            try:
                result = future.result()
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
                
                results.append(PrimitiveResult(
                    primitive_id=job.primitive_id,
                    num_correct=0.0,
                    accuracy_score=0.0,
                    success=False,
                    error_msg=f"Worker exception: {str(e)[:100]}"
                ))
        
        return results

# Global pool instance (singleton pattern for notebook compatibility)
_global_pool: Optional[FastTransformPool] = None

def get_global_transform_pool(library=None) -> FastTransformPool:
    """Get or create global transform pool"""
    global _global_pool
    if _global_pool is None and library is not None:
        _global_pool = FastTransformPool(library)
        _global_pool.start()
    elif _global_pool and not _global_pool._executor:
        # Pool exists but is shutdown, restart it
        print("Restarting existing transform pool...")
        _global_pool.start()
    return _global_pool

def shutdown_global_pool():
    """Shutdown global pool"""
    global _global_pool
    if _global_pool:
        _global_pool.shutdown()
        _global_pool = None