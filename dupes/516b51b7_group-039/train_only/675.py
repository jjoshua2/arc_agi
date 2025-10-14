import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    output = grid.copy().astype(int)

    def is_solid_rect(component):
        if not component:
            return False
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        expected_size = (max_r - min_r + 1) * (max_c - min_c + 1)
        return len(component) == expected_size

    def fill_region(min_r, max_r, min_c, max_c, level):
        if min_r > max_r or min_c > max_c:
            return
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        current_color = level + 1
        if h <= 2 or w <= 2:
            fill_color = 2 if level >= 3 else current_color
            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    output[i, j] = fill_color
            return
        # Recurse inner first
        inner_min_r = min_r + 1
        inner_max_r = max_r - 1
        inner_min_c = min_c + 1
        inner_max_c = max_c - 1
        fill_region(inner_min_r, inner_max_r, inner_min_c, inner_max_c, level + 1)
        # Set border
        # Top and bottom
        for j in range(min_c, max_c + 1):
            output[min_r, j] = current_color
            output[max_r, j] = current_color
        # Sides in middle rows
        for i in range(min_r + 1, max_r):
            output[i, min_c] = current_color
            output[i, max_c] = current_color

    # Find and process components
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1 and not visited[i, j]:
                component = []
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    r, c = q.popleft()
                    component.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                if is_solid_rect(component):
                    min_r = min(r for r, _ in component)
                    max_r = max(r for r, _ in component)
                    min_c = min(c for _, c in component)
                    max_c = max(c for _, c in component)
                    fill_region(min_r, max_r, min_c, max_c, 0)

    return output.tolist()