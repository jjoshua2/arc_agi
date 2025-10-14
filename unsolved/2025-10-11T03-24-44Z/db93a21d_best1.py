from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Find all connected components of 9s
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 9 and not visited[r][c]:
                # BFS to find component bounds
                min_r, max_r = r, r
                min_c, max_c = c, c
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    x, y = queue.popleft()
                    min_r = min(min_r, x)
                    max_r = max(max_r, x)
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 9:
                            visited[nx][ny] = True
                            queue.append((nx, ny))
                components.append((min_r, max_r, min_c, max_c))
    
    # Compute expanded bounds for each component
    expanded_comps = []
    for min_r, max_r, min_c, max_c in components:
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        t = min(h, w) // 2
        up_e = min(t, min_r)
        down_e = min(t, rows - 1 - max_r)
        left_e = min(t, min_c)
        right_e = min(t, cols - 1 - max_c)
        exp_minr = min_r - up_e
        exp_maxr = max_r + down_e
        exp_minc = min_c - left_e
        exp_maxc = max_c + right_e
        expanded_comps.append((min_r, max_r, min_c, max_c, exp_minr, exp_maxr, exp_minc, exp_maxc))
    
    # Place greens
    for _, _, _, _, exp_minr, exp_maxr, exp_minc, exp_maxc in expanded_comps:
        for rr in range(exp_minr, exp_maxr + 1):
            for cc in range(exp_minc, exp_maxc + 1):
                if grid[rr][cc] == 0:
                    output[rr][cc] = 3
    
    # Sort components by original min_r
    expanded_comps.sort(key=lambda x: x[0])
    n = len(expanded_comps)
    
    # Place blues
    for i in range(n):
        min_r, max_r, min_c, max_c, exp_minr, exp_maxr, _, _ = expanded_comps[i]
        start_r = exp_maxr + 1
        if start_r > rows - 1:
            continue
        if i == n - 1:
            end_r = rows - 1
        else:
            next_min_r, next_max_r, _, _, next_exp_minr, _, _, _ = expanded_comps[i + 1]
            if next_min_r > max_r:
                potential_end = next_exp_minr
                if potential_end >= start_r:
                    end_r = potential_end
                else:
                    end_r = rows - 1
            else:
                end_r = rows - 1
        for r in range(start_r, end_r + 1):
            for c in range(min_c, max_c + 1):
                if output[r][c] == 0:
                    output[r][c] = 1
    
    return output