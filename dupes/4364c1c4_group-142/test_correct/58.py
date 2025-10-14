def find_components(grid, background):
    h = len(grid)
    w = len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] != background and not visited[i][j]:
                color = grid[i][j]
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                min_r = i
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    min_r = min(min_r, x)
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append((min_r, component, color))
    return components

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    w = len(grid[0])
    background = grid[0][0]
    components = find_components(grid, background)
    components.sort(key=lambda x: x[0])
    out_grid = [row[:] for row in grid]
    for idx, (min_r, comp, color) in enumerate(components):
        direction = -1 if idx % 2 == 0 else 1
        row_cols = {}
        for r, c in comp:
            if r not in row_cols:
                row_cols[r] = set()
            row_cols[r].add(c)
        for r in row_cols:
            cols = sorted(row_cols[r])
            if not cols:
                continue
            segments = []
            start = cols[0]
            current = cols[0]
            for k in range(1, len(cols)):
                if cols[k] == current + 1:
                    current = cols[k]
                else:
                    segments.append((start, current))
                    start = cols[k]
                    current = cols[k]
            segments.append((start, current))
            for left, right in segments:
                new_left = left + direction
                new_right = right + direction
                # Set the new range clipped
                start_j = max(0, min(new_left, new_right))
                end_j = min(w - 1, max(new_left, new_right))
                for j in range(start_j, end_j + 1):
                    out_grid[r][j] = color
                # Trim old positions not in new range
                old_start = min(left, right)
                old_end = max(left, right)
                for j in range(old_start, old_end + 1):
                    new_min = min(new_left, new_right)
                    new_max = max(new_left, new_right)
                    if not (new_min <= j <= new_max):
                        out_grid[r][j] = background
    return out_grid