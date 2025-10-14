import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    purple_color = 8
    rows, cols = grid.shape

    # Find all connected components of purple
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_solid_rect(component):
        if not component:
            return False, None, None, None, None
        rs = [r for r, c in component]
        cs = [c for r, c in component]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        expected_size = (max_r - min_r + 1) * (max_c - min_c + 1)
        if len(component) != expected_size:
            return False, None, None, None, None
        # Check all cells in bbox are purple
        for i in range(min_r, max_r + 1):
            for j in range(min_c, max_c + 1):
                if grid[i, j] != purple_color:
                    return False, None, None, None, None
        return True, min_r, max_r, min_c, max_c

    output = grid.copy()

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == purple_color and not visited[i, j]:
                # BFS to find component
                component = []
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    r, c = q.popleft()
                    component.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == purple_color and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                is_rect, min_r, max_r, min_c, max_c = is_solid_rect(component)
                if not is_rect:
                    continue
                # Now compute closest for each border position
                # Top border
                for j in range(min_c, max_c + 1):
                    for k in range(min_r - 1, -1, -1):  # from just above down to 0
                        if grid[k, j] != 0:
                            output[min_r, j] = grid[k, j]
                            break
                # Bottom border
                for j in range(min_c, max_c + 1):
                    for k in range(max_r + 1, rows):
                        if grid[k, j] != 0:
                            output[max_r, j] = grid[k, j]
                            break
                # Left border
                for i in range(min_r, max_r + 1):
                    for jj in range(min_c - 1, -1, -1):  # from just left to 0
                        if grid[i, jj] != 0:
                            output[i, min_c] = grid[i, jj]
                            break
                # Right border
                for i in range(min_r, max_r + 1):
                    for jj in range(max_c + 1, cols):
                        if grid[i, jj] != 0:
                            output[i, max_c] = grid[i, jj]
                            break

    return output.tolist()