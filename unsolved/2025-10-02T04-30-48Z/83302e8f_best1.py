import numpy as np
from collections import deque, Counter

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return [row[:] for row in grid_lst]
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    non_zero_vals = grid[grid > 0]
    if len(non_zero_vals) == 0:
        return grid.tolist()
    unique_non_zero = np.unique(non_zero_vals)
    if len(unique_non_zero) == 1:
        # Fill mode
        L = unique_non_zero[0]
        visited = np.zeros((rows, cols), dtype=bool)
        q = deque()
        # Add border 0s to queue
        for i in range(rows):
            for j in [0, cols - 1]:
                if grid[i, j] == 0 and not visited[i, j]:
                    q.append((i, j))
                    visited[i, j] = True
                    grid[i, j] = 3
        for j in range(cols):
            for i in [0, rows - 1]:
                if grid[i, j] == 0 and not visited[i, j]:
                    q.append((i, j))
                    visited[i, j] = True
                    grid[i, j] = 3
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            x, y = q.popleft()
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0 and not visited[nx, ny]:
                    visited[nx, ny] = True
                    grid[nx, ny] = 3
                    q.append((nx, ny))
        # Set remaining 0s to 4
        grid[grid == 0] = 4
    else:
        # Injection mode
        counts = Counter(non_zero_vals)
        L = max(counts, key=counts.get)
        struct = (grid == L)
        for i in range(rows):
            for j in range(cols):
                col = grid[i, j]
                if col > 0 and col != L:
                    changed = False
                    # Check same row
                    row_struct_cols = np.where(struct[i, :])[0]
                    if len(row_struct_cols) > 0:
                        minc = row_struct_cols.min()
                        maxc = row_struct_cols.max()
                        if j < minc:
                            pos = i * cols + minc
                            if grid[i, minc] == L:
                                grid[i, minc] = col
                            changed = True
                        elif j > maxc:
                            pos = i * cols + maxc
                            if grid[i, maxc] == L:
                                grid[i, maxc] = col
                            changed = True
                    # If not changed, check same column
                    if not changed:
                        col_struct_rows = np.where(struct[:, j])[0]
                        if len(col_struct_rows) > 0:
                            minr = col_struct_rows.min()
                            maxr = col_struct_rows.max()
                            if i < minr:
                                if grid[minr, j] == L:
                                    grid[minr, j] = col
                                changed = True
                            elif i > maxr:
                                if grid[maxr, j] == L:
                                    grid[maxr, j] = col
                                changed = True
    return grid.tolist()