from collections import deque
import copy

def get_neighbors_4(r, c, h, w):
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    res = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            res.append((nr, nc))
    return res

def get_neighbors_8(r, c, h, w):
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    res = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            res.append((nr, nc))
    return res

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = grid_lst
    h = len(grid)
    w = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find all connected components of 1's using 4-connectivity
    visited = [[False] * w for _ in range(h)]
    components = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 1 and not visited[i][j]:
                comp = []
                queue = deque([(i, j)])
                visited[i][j] = True
                while queue:
                    r, c = queue.popleft()
                    comp.append((r, c))
                    for nr, nc in get_neighbors_4(r, c, h, w):
                        if grid[nr][nc] == 1 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append(comp)
    
    for comp in components:
        if not comp:
            continue
        comp_set = set(comp)
        
        # Create temp grid with only this component as 1, others 0
        temp = [[0] * w for _ in range(h)]
        for r, c in comp:
            temp[r][c] = 1
        
        # Flood fill from borders to mark outside (4-connectivity)
        visited_temp = [[False] * w for _ in range(h)]
        queue = deque()
        # Left and right borders
        for i in range(h):
            for j in [0, w - 1]:
                if temp[i][j] == 0 and not visited_temp[i][j]:
                    queue.append((i, j))
                    visited_temp[i][j] = True
        # Top and bottom borders
        for j in range(w):
            for i in [0, h - 1]:
                if temp[i][j] == 0 and not visited_temp[i][j]:
                    queue.append((i, j))
                    visited_temp[i][j] = True
        while queue:
            r, c = queue.popleft()
            for nr, nc in get_neighbors_4(r, c, h, w):
                if temp[nr][nc] == 0 and not visited_temp[nr][nc]:
                    visited_temp[nr][nc] = True
                    queue.append((nr, nc))
        
        # Check if there is any inside 0
        has_inside = False
        for i in range(h):
            for j in range(w):
                if temp[i][j] == 0 and not visited_temp[i][j]:
                    has_inside = True
                    break
            if has_inside:
                break
        if not has_inside:
            continue
        
        # Now color adjacent original 0 cells
        for i in range(h):
            for j in range(w):
                if grid[i][j] != 0:
                    continue  # only original 0's
                # Check if adjacent to any in comp via 8-neigh
                adj = False
                for ni, nj in get_neighbors_8(i, j, h, w):
                    if (ni, nj) in comp_set:
                        adj = True
                        break
                if adj:
                    # Determine if inside: temp[i][j]==0 (always here) and not visited_temp[i][j]
                    if not visited_temp[i][j]:
                        output[i][j] = 3
                    else:
                        output[i][j] = 2
    
    return output