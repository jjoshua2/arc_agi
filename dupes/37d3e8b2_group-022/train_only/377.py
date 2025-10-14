from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    H = len(grid)
    W = len(grid[0])
    
    # Label 8 components
    label = [[0] * W for _ in range(H)]
    visited_comp = [[False] * W for _ in range(H)]
    comp_id = 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 8 and not visited_comp[r][c]:
                q = deque([(r, c)])
                visited_comp[r][c] = True
                label[r][c] = comp_id
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and not visited_comp[nx][ny] and grid[nx][ny] == 8:
                            visited_comp[nx][ny] = True
                            label[nx][ny] = comp_id
                            q.append((nx, ny))
                comp_id += 1
    
    num_comp = comp_id - 1
    hole_counts = [0] * (num_comp + 1)
    
    # Flood fill external 0s from borders
    visited_bg = [[False] * W for _ in range(H)]
    q = deque()
    for r in range(H):
        for c in range(W):
            if (r == 0 or r == H - 1 or c == 0 or c == W - 1) and grid[r][c] == 0:
                if not visited_bg[r][c]:
                    visited_bg[r][c] = True
                    q.append((r, c))
    
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == 0 and not visited_bg[nx][ny]:
                visited_bg[nx][ny] = True
                q.append((nx, ny))
    
    # Now find internal hole components
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0 and not visited_bg[r][c]:
                # New internal hole
                hole_comp = []
                q = deque([(r, c)])
                visited_bg[r][c] = True
                hole_comp.append((r, c))
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == 0 and not visited_bg[nx][ny]:
                            visited_bg[nx][ny] = True
                            q.append((nx, ny))
                            hole_comp.append((nx, ny))
                
                # Find surrounding component ids
                surrounding = set()
                for hx, hy in hole_comp:
                    for dx, dy in directions:
                        sx, sy = hx + dx, hy + dy
                        if 0 <= sx < H and 0 <= sy < W and grid[sx][sy] == 8:
                            surrounding.add(label[sx][sy])
                
                if len(surrounding) == 1:
                    the_id = next(iter(surrounding))
                    hole_counts[the_id] += 1
    
    # Create output
    out = [row[:] for row in grid]
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 8:
                cid = label[r][c]
                out[r][c] = hole_counts[cid]
    
    return out