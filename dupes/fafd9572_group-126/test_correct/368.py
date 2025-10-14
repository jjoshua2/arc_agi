from typing import List, Tuple

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return [row[:] for row in grid_lst]
    
    h = len(grid_lst)
    w = len(grid_lst[0])
    
    # Collect hints: positions with value >= 2
    hints: List[Tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid_lst[r][c] >= 2:
                hints.append((r, c, grid_lst[r][c]))
    
    # Sort hints by row, then column
    hints.sort()
    hints_colors = [col for _, _, col in hints]
    
    # Directions for 8-connectivity
    directions = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)]
    
    # Visited matrix
    visited = [[False] * w for _ in range(h)]
    
    # Find components
    components: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if grid_lst[r][c] == 1 and not visited[r][c]:
                # DFS to find component
                comp: List[Tuple[int, int]] = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr, dc in directions:
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid_lst[nr][nc] == 1:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                
                if comp:
                    min_r = min(rr for rr, _ in comp)
                    min_c = min(cc for _, cc in comp)
                    components.append((min_r, min_c, comp))
    
    # Sort components by min_r, then min_c
    components.sort()
    
    # Create output grid
    output = [row[:] for row in grid_lst]
    
    # Assign colors
    for i, (_, _, comp) in enumerate(components):
        if i < len(hints_colors):
            new_color = hints_colors[i]
            for rr, cc in comp:
                output[rr][cc] = new_color
    
    return output