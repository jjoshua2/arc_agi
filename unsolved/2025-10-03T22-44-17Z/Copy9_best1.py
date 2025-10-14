from collections import deque
from typing import List, Tuple

def get_components(grid: List[List[int]], C: int) -> List[List[Tuple[int, int]]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == C and not visited[i][j]:
                comp = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    r, c = q.popleft()
                    comp.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == C:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)
    return components

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    templates = {}  # C -> (min_r, max_r, min_c, max_c, center_r, center_c)
    indicators = {}  # C -> list of (ir, ic)
    
    for C in range(1, 10):
        comps = get_components(grid, C)
        templ = None
        templ_size = 0
        inds = []
        for comp in comps:
            if len(comp) == 1:
                r, c = comp[0]
                inds.append((r, c))
            elif len(comp) > 1:
                size = len(comp)
                if size > templ_size:
                    templ_size = size
                    min_r = min(r for r, c in comp)
                    max_r = max(r for r, c in comp)
                    min_c = min(c for r, c in comp)
                    max_c = max(c for r, c in comp)
                    center_r = (min_r + max_r) // 2
                    center_c = (min_c + max_c) // 2
                    templ = (min_r, max_r, min_c, max_c, center_r, center_c)
        if templ:
            templates[C] = templ
        if inds:
            indicators[C] = inds
    
    for C in range(1, 10):
        if C in templates and C in indicators:
            min_r, max_r, min_c, max_c, center_r, center_c = templates[C]
            for ir, ic in indicators[C]:
                offset_r = ir - center_r
                offset_c = ic - center_c
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        val = grid[rr][cc]
                        nr = rr + offset_r
                        nc = cc + offset_c
                        if 0 <= nr < rows and 0 <= nc < cols:
                            output[nr][nc] = val
    
    return output