from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    components: List[Tuple[int, int, int, int]] = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in visited:
                color = grid[r][c]
                queue = deque([(r, c)])
                visited.add((r, c))
                min_r_comp = max_r_comp = r
                min_c_comp = c
                # No need for max_c_comp
                
                while queue:
                    cr, cc = queue.popleft()
                    min_r_comp = min(min_r_comp, cr)
                    max_r_comp = max(max_r_comp, cr)
                    min_c_comp = min(min_c_comp, cc)
                    
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                            grid[nr][nc] == color and (nr, nc) not in visited):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                
                height = max_r_comp - min_r_comp + 1
                components.append((color, height, min_c_comp, min_r_comp))
    
    if not components:
        return []
    
    # Sort: descending height, ascending min_c, ascending min_r
    components.sort(key=lambda x: (-x[1], x[2], x[3]))
    
    max_h = max(comp[1] for comp in components)
    num_cols_out = len(components)
    output = [[0] * num_cols_out for _ in range(max_h)]
    
    for col_idx, (color, h, _, _) in enumerate(components):
        for row_idx in range(h):
            output[row_idx][col_idx] = color
    
    return output