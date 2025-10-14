from collections import deque
import typing

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    
    total_cc = 0
    
    for c in range(1, 10):
        positions: list[tuple[int, int]] = [(r, cc) for r in range(h) for cc in range(w) if grid[r][cc] == c]
        if not positions:
            continue
        
        cols_used = set(cc for _, cc in positions)
        if len(cols_used) == 1:
            col_x = next(iter(cols_used))
            rows_with_c = sorted(set(r for r, _ in positions))
            if not rows_with_c:
                continue
            num_gen = 1
            for i in range(1, len(rows_with_c)):
                if rows_with_c[i] - rows_with_c[i - 1] > 2:
                    num_gen += 1
            total_cc += num_gen
        else:
            # Standard 4-connected CC count for color c
            visited = set()
            cc_count = 0
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for r, cc in positions:
                if (r, cc) in visited:
                    continue
                # BFS
                queue = deque([(r, cc)])
                visited.add((r, cc))
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < h and 0 <= ny < w and
                            grid[nx][ny] == c and (nx, ny) not in visited):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                cc_count += 1
            total_cc += cc_count
    
    if total_cc == 0:
        return [[]]
    return [[0] * total_cc]