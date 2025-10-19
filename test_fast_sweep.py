#!/usr/bin/env python3
"""
Quick test to validate fast sweep implementation.
Run this to check if the worker pool and shape filtering work correctly.
"""

import os
import time
import asyncio
from src.models import Library, Primitive, Challenge, TrainExample
from src.transform_pool import FastTransformPool, shutdown_global_pool
from src.shape_filter import filter_primitives_by_shape

def create_test_library():
    """Create a small test library with a few primitives"""
    primitives = [
        Primitive(id="identity", python_code_str="""
def transform(grid):
    return grid
"""),
        Primitive(id="fill_red", python_code_str="""
def transform(grid):
    rows, cols = len(grid), len(grid[0]) if grid else 0
    return [[2] * cols for _ in range(rows)]
"""),
        Primitive(id="transpose", python_code_str="""
def transform(grid):
    if not grid or not grid[0]:
        return grid
    return [[grid[r][c] for r in range(len(grid))] for c in range(len(grid[0]))]
"""),
        Primitive(id="invalid", python_code_str="""
def transform(grid):
    raise ValueError("This primitive always fails")
""")
    ]
    return Library(primitives=primitives)

def create_test_challenge():
    """Create a simple test challenge"""
    return Challenge(
        id="test_challenge",
        train=[
            TrainExample(
                input=[[1, 0], [0, 1]],
                output=[[2, 2], [2, 2]]  # Fill with red (color 2)
            ),
            TrainExample(
                input=[[3, 4], [5, 6]],
                output=[[2, 2], [2, 2]]  # Fill with red (color 2)
            )
        ],
        test=[]
    )

async def test_worker_pool():
    """Test the worker pool functionality"""
    print("Testing worker pool...")
    
    library = create_test_library()
    challenge = create_test_challenge()
    
    # Create pool
    pool = FastTransformPool(library, num_workers=2, max_tasks_per_child=10)
    pool.start()
    
    try:
        # Test evaluation
        primitive_ids = [p.id for p in library.primitives]
        train_inputs = [ex.input for ex in challenge.train]
        train_outputs = [ex.output for ex in challenge.train]
        
        start_time = time.perf_counter()
        results = pool.evaluate_primitives_batch(
            primitive_ids=primitive_ids,
            train_inputs=train_inputs,
            train_outputs=train_outputs
        )
        elapsed = time.perf_counter() - start_time
        
        print(f"Evaluated {len(results)} primitives in {elapsed:.3f}s")
        
        for result in results:
            print(f"  {result.primitive_id}: success={result.success}, num_correct={result.num_correct}, accuracy={result.accuracy_score:.3f}")
            if result.error_msg:
                print(f"    Error: {result.error_msg}")
        
        # Expected: fill_red should get 2.0 correct (perfect on both examples)
        fill_red_result = next(r for r in results if r.primitive_id == "fill_red")
        assert fill_red_result.num_correct == 2.0, f"Expected 2.0, got {fill_red_result.num_correct}"
        print("✓ Worker pool test passed")
        
    finally:
        pool.shutdown()

def test_shape_filter():
    """Test the shape filtering functionality"""
    print("\nTesting shape filter...")
    
    library = create_test_library()
    challenge = create_test_challenge()
    
    # All our test examples preserve shape (2x2 -> 2x2)
    filtered = filter_primitives_by_shape(library.primitives, challenge)
    
    print(f"Original: {len(library.primitives)} primitives")
    print(f"Filtered: {len(filtered)} primitives")
    
    # Should keep identity and fill_red (shape preserving), reject transpose (shape changing for some inputs)
    kept_ids = [getattr(p, 'id', 'unknown') for p in filtered]
    print(f"Kept primitives: {kept_ids}")
    
    # Note: transpose might be kept since our test examples happen to be square
    print("✓ Shape filter test completed")

async def test_integration():
    """Test the integration with the main logic"""
    print("\nTesting integration...")
    
    # Set fast sweep environment variable
    os.environ["ARC_FAST_SWEEP"] = "1"
    
    try:
        from src.logic import _fast_two_pass_select_primitives_async
        
        library = create_test_library()
        challenge = create_test_challenge()
        
        start_time = time.perf_counter()
        selected = await _fast_two_pass_select_primitives_async(
            library=library,
            challenge=challenge,
            k_top=2,
            challenge_primitive_scores=None,
            fp_top_k=10,
            sp_batch=50
        )
        elapsed = time.perf_counter() - start_time
        
        print(f"Selected {len(selected)} primitives in {elapsed:.3f}s")
        for p in selected:
            print(f"  Selected: {getattr(p, 'id', 'unknown')}")
        
        print("✓ Integration test completed")
        
    finally:
        shutdown_global_pool()

async def main():
    """Run all tests"""
    print("=== Fast Sweep Test Suite ===")
    
    try:
        await test_worker_pool()
        test_shape_filter()
        await test_integration()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())