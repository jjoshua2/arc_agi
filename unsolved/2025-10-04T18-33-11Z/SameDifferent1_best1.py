from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]  # make a copy
    
    # Find all connected components of non-zero cells
    components = []
    visited = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in visited:
                comp = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in comp:
                        continue
                    comp.add((cr, cc))
                    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] != 0 and (nr, nc) not in comp:
                            stack.append((nr, nc))
                visited.update(comp)
                components.append(comp)
    
    # For each component, compute the bounding box and area
    comp_areas = []
    for comp in components:
        min_r = min(r for (r, c) in comp)
        max_r = max(r for (r, c) in comp)
        min_c = min(c for (r, c) in comp)
        max_c = max(c for (r, c) in comp)
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        area = width * height
        comp_areas.append(area)
    
    if not comp_areas:
        return result  # no non-zero cells
    
    min_area = min(comp_areas)
    
    # For each component, if its area is greater than min_area, remove it (set to 0)
    for idx, comp in enumerate(components):
        if comp_areas[idx] > min_area:
            for (r, c) in comp:
                result[r][c] = 0
                
    return result