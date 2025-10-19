# Fast Sweep Optimization

This implements a fast primitive evaluation system to speed up the initial "library sweep" that was taking 1000+ seconds per challenge.

## What it does

- **Persistent Worker Pool**: Replaces per-primitive subprocess spawning with reusable worker processes
- **Shape-based Filtering**: Quickly rejects primitives that can't possibly work based on input/output shape patterns
- **Inline Execution**: Runs transform functions directly in worker memory instead of subprocess + temp files

## Expected speedup

Target: **3-5x faster** first pass (from ~1151s to ~200-400s) while preserving identical results.

## Usage

### Enable/Disable
```bash
# Enable fast sweep (default)
export ARC_FAST_SWEEP=1

# Disable fast sweep (fallback to original)
export ARC_FAST_SWEEP=0
```

### Configuration
```bash
# Worker pool settings (defaults shown)
export ARC_FAST_SWEEP_WORKERS=4        # Number of worker processes
export ARC_FAST_SWEEP_BATCH_SIZE=100   # Primitives per batch
export ARC_FAST_SWEEP_TASKS_PER_CHILD=200  # Restart workers after N tasks
```

### Testing
```bash
# Run the test suite
python test_fast_sweep.py
```

### Integration with existing runs
The fast sweep automatically integrates with existing code paths:
- `python -m src.submission` (production path)
- `python -m src.main` (development path)

## Technical details

### Shape filtering rules
- **Shape-preserving primitives** (fill, color, replace): Rejected if ANY training example changes shape
- **Shape-changing primitives** (crop, rotate, transpose): Rejected if ALL training examples preserve shape
- **Size ratio check**: Reject if output is >10x or <0.1x input area (likely errors)

### Worker pool design
- Uses `ProcessPoolExecutor` with `forkserver` context on Linux (Kaggle-compatible)
- Workers cache the entire library on startup to avoid repeated deserialization
- Inline execution: `exec()` the transform code directly instead of subprocess
- Automatic cleanup on program exit

### Fallback safety
- If fast sweep fails for any reason, automatically falls back to original implementation
- Preserves identical primitive selection and scoring logic
- No impact on LLM calls, second pass, or final results

## Monitoring

Look for these log messages:
```
[challenge_id] Shape filter: kept 1234/2064 primitives (rejected 830)
[challenge_id] Fast sweep: found 12 perfect-on-first primitives  
[challenge_id] Fast sweep: shape=0.1s, first=45.2s, second=8.1s, total=53.4s
Started transform pool: 4 workers, forkserver context
Cleaned up transform pool
```

## Kaggle notebook tips

1. **Avoid cell re-execution**: Create the pool once at the notebook level, don't recreate in frequently re-run cells
2. **Memory management**: Workers are restarted every 200 tasks by default to prevent memory leaks
3. **Context choice**: Uses `forkserver` by default on Linux for notebook compatibility

## Troubleshooting

### Common issues
- **Import errors**: Make sure `src/transform_pool.py` and `src/shape_filter.py` are in your path
- **Multiprocessing errors**: Try `ARC_FAST_SWEEP=0` to disable and compare performance
- **Memory issues**: Reduce `ARC_FAST_SWEEP_WORKERS` or `ARC_FAST_SWEEP_BATCH_SIZE`

### Performance debugging
The system logs timing for each phase. If first pass is still slow:
1. Check if shape filtering is rejecting enough primitives
2. Verify worker pool is starting correctly 
3. Look for error messages in primitive evaluation
4. Compare with original timing using `ARC_FAST_SWEEP=0`

### Validation
Run a few challenges with both `ARC_FAST_SWEEP=1` and `ARC_FAST_SWEEP=0` and verify:
- Same primitives are selected (same IDs in same order)
- Same scores and rankings
- Faster wall clock time with fast sweep enabled