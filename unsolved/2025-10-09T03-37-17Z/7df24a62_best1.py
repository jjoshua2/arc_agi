import numpy as np

def get_filled_components(grid):
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] in (1, 4) and not visited[r, c]:
                component = []
                stack = [(r, c)]
                visited[r, c] = True
                min_r, max_r, min_c, max_c = r, r, c, c
                
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                            grid[nr, nc] in (1, 4) and not visited[nr, nc]):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                area = height * width
                is_square = height == width
                is_filled = len(component) == area and is_square
                
                if is_filled:
                    num_fours = sum(1 for rr, cc in component if grid[rr, cc] == 4)
                    four_rel = [(rr - min_r, cc - min_c) for rr, cc in component if grid[rr, cc] == 4]
                    components.append((height, num_fours, min_r, min_c, four_rel))
    
    return components

def apply_transform(pos_list, s, flip, rot):
    current = pos_list[:]
    if flip:
        current = [(r, s - 1 - c) for r, c in current]
    for _ in range(rot):
        current = [(c, s - 1 - r) for r, c in current]
    return frozenset(current)

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    components = get_filled_components(grid)
    if not components:
        return grid.tolist()
    
    # Select template: max s, then max num_fours
    components.sort(key=lambda x: (x[0], x[1]), reverse=True)
    s, _, min_r, min_c, four_pos = components[0]
    
    # Generate 8 symmetry patterns
    patterns = []
    for flip in [False, True]:
        for rot in range(4):
            pat = apply_transform(four_pos, s, flip, rot)
            patterns.append(pat)
    
    # Find valid placements using input grid
    placements = []
    for r0 in range(rows - s + 1):
        for c0 in range(cols - s + 1):
            for expected in patterns:
                is_match = True
                has_extra = False
                all_valid = True
                for i in range(s):
                    if not is_match or has_extra or not all_valid:
                        break
                    for j in range(s):
                        val = grid[r0 + i, c0 + j]
                        rel = (i, j)
                        if rel in expected:
                            if val != 4:
                                is_match = False
                                break
                        else:
                            if val == 4:
                                has_extra = True
                                break
                            if val not in (0, 1):
                                all_valid = False
                                break
                if is_match and not has_extra and all_valid:
                    placements.append((r0, c0))
                    break  # One symmetry per position; no need for multiples
    
    # Apply fills
    result = grid.copy()
    for r0, c0 in placements:
        for i in range(s):
            for j in range(s):
                if result[r0 + i, c0 + j] == 0:
                    result[r0 + i, c0 + j] = 1
    
    return result.tolist()