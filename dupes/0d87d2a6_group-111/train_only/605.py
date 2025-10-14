from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    direct_fills: set[Tuple[int, int]] = set()
    
    # Find rows with >=2 ones and mark horizontal fills
    for i in range(rows):
        ones_cols = [j for j in range(cols) if grid[i][j] == 1]
        if len(ones_cols) >= 2:
            min_c = min(ones_cols)
            max_c = max(ones_cols)
            for c in range(min_c, max_c + 1):
                direct_fills.add((i, c))
    
    # Find columns with >=2 ones and mark vertical fills
    for j in range(cols):
        ones_rows = [i for i in range(rows) if grid[i][j] == 1]
        if len(ones_rows) >= 2:
            min_r = min(ones_rows)
            max_r = max(ones_rows)
            for r in range(min_r, max_r + 1):
                direct_fills.add((r, j))
    
    # Find seed positions: direct fills that are 2 in input
    seeds = [(r, c) for r, c in direct_fills if grid[r][c] == 2]
    
    # Multi-source BFS to find all connected 2's from seeds (4-connected)
    to_be_one: set[Tuple[int, int]] = set()
    if seeds:
        queue = deque(seeds)
        visited = set(seeds)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            r, c = queue.popleft()
            to_be_one.add((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr][nc] == 2 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    # Set all connected 2's to 1 in output
    for r, c in to_be_one:
        output[r][c] = 1
    
    # Set direct fills that are not 2 to 1
    for r, c in direct_fills:
        if grid[r][c] != 2:
            output[r][c] = 1
    
    return output