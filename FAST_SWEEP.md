# Fast Sweep Optimization

This implements a fast primitive evaluation system to speed up the initial "library sweep" that was taking 1000+ seconds per challenge.

## What it does

- **Persistent Worker Pool**: Replaces per-primitive subprocess spawning with reusable worker processes
- **Primitive Blocklist**: Automatically tracks and blocks primitives that consistently crash workers
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
Loaded primitive blocklist: 15 blocked primitives
🚫 Filtered out 15 blocked primitives (2049/2064 remaining)
[challenge_id] Shape filter: kept 1234/2049 primitives (rejected 815)
[challenge_id] Fast sweep: found 12 perfect-on-first primitives
⚠️  Primitive bad_func_123 failed (2/3 to block, memoryerror)
🚫 BLOCKED primitive bad_func_456 after 3 failures in 60.0m (systemerror)
[challenge_id] Fast sweep: blocklist=0.001s, shape=0.001s, first=45.2s, second=8.1s, total=45.3s
Started transform pool: 4 workers, forkserver context
Cleaned up transform pool

============================================================
🚫 PRIMITIVE REMOVAL REPORT
============================================================
Total primitives blocked: 23
Total failures recorded: 89
Estimated crashes prevented: 67
Time saved by avoiding known-bad primitives: ~33.5 seconds

📋 REMOVED PRIMITIVES (sorted by failure count):
------------------------------------------------------------
ID                   Failures Recent First Seen Last Seen 
------------------------------------------------------------
bad_func_456         8        3      2.1h       15.2m     
memory_hog           6        2      3.5h       45.1m     
crash_generator      5        1      1.2h       120.5m    
... and 20 more blocked primitives

🔥 WORST OFFENDERS (total failures):
  1. bad_func_456: 8 failures
  2. memory_hog: 6 failures
  3. crash_generator: 5 failures

💡 RECOMMENDATIONS:
  • Library quality issue: 67 crashes prevented
  • Review primitive generation process to reduce error-prone functions

📁 Detailed logs saved to: primitive_blocklist.json
   Use 'python manage_blocklist.py export' for analysis
============================================================
```

## Kaggle notebook tips

1. **Avoid cell re-execution**: Create the pool once at the notebook level, don't recreate in frequently re-run cells
2. **Memory management**: Workers are restarted every 200 tasks by default to prevent memory leaks
3. **Context choice**: Uses `forkserver` by default on Linux for notebook compatibility

## Blocklist Management

Problematic primitives are automatically blocked after 3 failures within 1 hour. The blocklist persists across runs in `primitive_blocklist.json`.

### Manual blocklist management
```bash
# Check blocklist status
python manage_blocklist.py status

# List all blocked primitives
python manage_blocklist.py list --verbose

# Manually block a primitive
python manage_blocklist.py block bad_primitive_id --reason="causes crashes"

# Unblock a primitive for testing
python manage_blocklist.py unblock primitive_id

# Clear entire blocklist
python manage_blocklist.py clear --confirm

# Export blocklist data for analysis
python manage_blocklist.py export analysis.json
```

## Troubleshooting

### Common issues
- **Import errors**: Make sure `src/transform_pool.py`, `src/shape_filter.py`, and `src/primitive_blocklist.py` are in your path
- **Multiprocessing errors**: Try `ARC_FAST_SWEEP=0` to disable and compare performance
- **Memory issues**: Reduce `ARC_FAST_SWEEP_WORKERS` or `ARC_FAST_SWEEP_BATCH_SIZE`
- **Too many blocked primitives**: Check `python manage_blocklist.py status` and consider clearing old blocks

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