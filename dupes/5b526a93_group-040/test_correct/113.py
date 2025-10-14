import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find all blue pluses: v -> set of h
    plus_positions = {}
    pattern_pos = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]
    for v in range(rows - 2):
        for h in range(cols - 2):
            # Check if blue plus
            is_plus = True
            for dr, dc in pattern_pos:
                if grid[v + dr, h + dc] != 1:
                    is_plus = False
                    break
            if is_plus and grid[v + 1, h + 1] == 0:
                if v not in plus_positions:
                    plus_positions[v] = set()
                plus_positions[v].add(h)
    
    if not plus_positions:
        return grid.tolist()
    
    # Find max count
    max_count = max(len(hs) for hs in plus_positions.values())
    
    # Find template H from a v with max_count
    template_vs = [v for v, hs in plus_positions.items() if len(hs) == max_count]
    template_v = min(template_vs)  # arbitrary, assume consistent
    H = plus_positions[template_v]
    
    # For each v with pluses, add missing purple pluses
    for v, existing_hs in plus_positions.items():
        for target_h in H:
            if target_h not in existing_hs:
                # Place purple plus
                for dr, dc in pattern_pos:
                    grid[v + dr, target_h + dc] = 8
    
    return grid.tolist()