from collections import deque, defaultdict
from typing import List, Tuple

def find_regions(grid: List[List[int]], color: int) -> List[List[Tuple[int, int]]]:
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    regions = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == color and not visited[i][j]:
                region = []
                queue = deque([(i, j)])
                visited[i][j] = True
                while queue:
                    ci, cj = queue.popleft()
                    region.append((ci, cj))
                    for di, dj in directions:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == color and not visited[ni][nj]:
                            visited[ni][nj] = True
                            queue.append((ni, nj))
                regions.append(region)
    return regions

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    
    total_cells = [0] * 10
    for i in range(rows):
        for j in range(cols):
            total_cells[grid[i][j]] += 1
    
    output = [row[:] for row in grid]
    
    adj_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for color in range(1, 10):
        regions = find_regions(grid, color)
        for region in regions:
            sz = len(region)
            if sz >= 3:
                continue
            elif sz == 2:
                for ri, ci in region:
                    output[ri][ci] = 0
            elif sz == 1:
                r, c = region[0]
                old_color = grid[r][c]
                
                adj_count = defaultdict(int)
                for dr, dc in adj_directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        adj_count[grid[nr][nc]] += 1
                
                candidates = []
                for T in list(adj_count.keys()):
                    if adj_count[T] < 2 or T == 0:
                        continue
                    
                    # Check outer relative to T
                    visited = set()
                    queue = deque()
                    # Add all border non-T
                    for i in range(rows):
                        for j in range(cols):
                            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and grid[i][j] != T:
                                if (i, j) not in visited:
                                    visited.add((i, j))
                                    queue.append((i, j))
                    
                    while queue:
                        ci, cj = queue.popleft()
                        for dr, dc in adj_directions:
                            ni, nj = ci + dr, cj + dc
                            if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] != T and (ni, nj) not in visited:
                                visited.add((ni, nj))
                                queue.append((ni, nj))
                    
                    is_outer = (r, c) in visited
                    if is_outer or adj_count[T] == 4:
                        candidates.append((adj_count[T], total_cells[T], T))
                
                if candidates:
                    # Choose the best: max count, then max total, then max T
                    best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
                    output[r][c] = best[2]
                else:
                    output[r][c] = 0
    
    return output