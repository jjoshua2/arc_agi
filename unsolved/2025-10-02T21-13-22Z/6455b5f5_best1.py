from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    h = len(grid)
    w = len(grid[0])
    visited = [[False for _ in range(w)] for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def is_adjacent_to_two(r: int, c: int) -> bool:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 2:
                return True
        return False

    def has_adjacent_two(comp: List[Tuple[int, int]]) -> bool:
        for r, c in comp:
            if is_adjacent_to_two(r, c):
                return True
        return False

    def touches_left(comp: List[Tuple[int, int]]) -> bool:
        for r, c in comp:
            if c == 0:
                return True
        return False

    def touches_top(comp: List[Tuple[int, int]]) -> bool:
        for r, c in comp:
            if r == 0:
                return True
        return False

    new_grid = [row[:] for row in grid]

    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not visited[i][j]:
                # BFS to find component
                component = []
                queue = deque([(i, j)])
                visited[i][j] = True
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                # Check if adjacent to 2
                if not has_adjacent_two(component):
                    continue  # Skip isolated 0s
                # Check touching conditions
                touch_left = touches_left(component)
                touch_top = touches_top(component)
                if touch_left and not touch_top:
                    continue  # Leave as 0
                # Fill based on size
                size = len(component)
                color = 8 if size <= 8 else 1
                for r, c in component:
                    new_grid[r][c] = color

    return new_grid