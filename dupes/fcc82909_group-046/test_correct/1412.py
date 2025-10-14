import numpy as np
from typing import List, Tuple

def get_components(grid: np.ndarray) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                component = []
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                components.append(component)
    return components

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = grid.copy()
    
    components = get_components(grid)
    
    for component in components:
        if not component:
            continue
        colors = set(grid[r, c] for r, c in component)
        K = len(colors)
        if K == 0:
            continue
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        
        start_r = max_r + 1
        for dr in range(K):
            r = start_r + dr
            if r >= rows:
                break
            for c in range(min_c, max_c + 1):
                if output[r, c] == 0:
                    output[r, c] = 3
    
    return output.tolist()