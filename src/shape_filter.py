"""
Shape-based primitive filtering for quick rejection.
Reduces the candidate set before expensive evaluation.
"""
import os
from typing import List, Set, Dict, Tuple
from src.models import Challenge, Primitive, GRID

def get_shape_signature(challenge: Challenge) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Extract input and output shapes for all training examples.
    
    Returns:
        (input_shapes, output_shapes) where each is list of (height, width) tuples
    """
    input_shapes = []
    output_shapes = []
    
    for train in challenge.train:
        input_shapes.append((len(train.input), len(train.input[0]) if train.input else 0))
        output_shapes.append((len(train.output), len(train.output[0]) if train.output else 0))
    
    return input_shapes, output_shapes

def should_reject_primitive_by_shape(primitive_id: str, input_shapes: List[Tuple[int, int]], output_shapes: List[Tuple[int, int]]) -> bool:
    """
    Quick shape-based rejection for known primitive categories.
    
    Args:
        primitive_id: ID of the primitive
        input_shapes: List of (height, width) for inputs
        output_shapes: List of (height, width) for outputs
        
    Returns:
        True if primitive should be rejected based on shape mismatch
    """
    # TEMPORARILY DISABLED: Shape filtering is rejecting all primitives
    # This needs investigation - primitive IDs might not match expected format
    return False
    
    # Conservative shape-based rules (only reject when we're very confident)
    
    # Check if all examples have same input->output shape pattern
    all_shapes_preserved = all(inp == out for inp, out in zip(input_shapes, output_shapes))
    any_shape_changed = any(inp != out for inp, out in zip(input_shapes, output_shapes))
    
    # Known shape-preserving primitives (conservative list)
    SHAPE_PRESERVING_KEYWORDS = [
        'fill', 'replace', 'color', 'paint', 'swap', 'change_color',
        'flood_fill', 'pattern_fill', 'recolor', 'substitute'
    ]
    
    # Known shape-changing primitives (conservative list) 
    SHAPE_CHANGING_KEYWORDS = [
        'crop', 'trim', 'extract', 'shrink', 'expand', 'resize',
        'rotate', 'transpose', 'flip', 'mirror', 'extend'
    ]
    
    primitive_lower = primitive_id.lower()
    
    # Rule 1: If primitive is known to preserve shapes, reject if ANY example changes shape
    if any(keyword in primitive_lower for keyword in SHAPE_PRESERVING_KEYWORDS):
        if any_shape_changed:
            return True
    
    # Rule 2: If primitive is known to change shapes, reject if ALL examples preserve shape
    if any(keyword in primitive_lower for keyword in SHAPE_CHANGING_KEYWORDS):
        if all_shapes_preserved:
            return True
    
    # Rule 3: Extreme size mismatches (likely errors)
    for inp_shape, out_shape in zip(input_shapes, output_shapes):
        inp_area = inp_shape[0] * inp_shape[1]
        out_area = out_shape[0] * out_shape[1]
        
        # Reject if output is >10x larger or <1/10th the input (likely malformed)
        if inp_area > 0 and out_area > 0:
            ratio = out_area / inp_area
            if ratio > 10.0 or ratio < 0.1:
                return True
    
    # Default: don't reject (be conservative)
    return False

def filter_primitives_by_shape(primitives: List[Primitive], challenge: Challenge) -> List[Primitive]:
    """
    Filter primitives using shape-based quick rejection.
    
    Args:
        primitives: List of primitives to filter
        challenge: Challenge with training examples
        
    Returns:
        Filtered list of primitives (subset of input)
    """
    input_shapes, output_shapes = get_shape_signature(challenge)
    
    # Debug shape information
    all_shapes_preserved = all(inp == out for inp, out in zip(input_shapes, output_shapes))
    any_shape_changed = any(inp != out for inp, out in zip(input_shapes, output_shapes))
    if os.environ.get("SUBMISSION_VERBOSE") == "1":
        print(f"[{challenge.id}] Challenge shapes: inputs={input_shapes}, outputs={output_shapes}")
        print(f"[{challenge.id}] Shape analysis: all_preserved={all_shapes_preserved}, any_changed={any_shape_changed}")
    
    kept = []
    rejected_count = 0
    rejected_reasons = []
    
    for primitive in primitives:
        primitive_id = getattr(primitive, 'id', 'unknown')
        
        if should_reject_primitive_by_shape(primitive_id, input_shapes, output_shapes):
            rejected_count += 1
            # Track first few rejection reasons for debugging
            if len(rejected_reasons) < 5:
                rejected_reasons.append(primitive_id)
        else:
            kept.append(primitive)
    
    if rejected_count > 0 and os.environ.get("SUBMISSION_VERBOSE") == "1":
        print(f"[{challenge.id}] Shape filter: kept {len(kept)}/{len(primitives)} primitives (rejected {rejected_count})")
        if rejected_reasons:
            print(f"[{challenge.id}] Sample rejected primitives: {rejected_reasons}")
    
    return kept
