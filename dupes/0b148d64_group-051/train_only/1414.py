import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape

    # Find the last all-zero row
    last_zero = -1
    for r in range(rows):
        if np.all(grid[r] == 0):
            last_zero = r

    bottom_start = last_zero + 1
    if bottom_start >= rows:
        return []

    # Now find connected components in bottom section using 4-connected flood fill
    visited = np.zeros((rows, cols), dtype=bool)
    components = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(bottom_start, rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                color = grid[r, c]
                stack = [(r, c)]
                visited[r, c] = True
                comp_size = 1
                min_r_comp = max_r_comp = r
                min_c_comp = max_c_comp = c
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                            nr >= bottom_start and not visited[nr, nc] and
                            grid[nr, nc] == color):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                            comp_size += 1
                            min_r_comp = min(min_r_comp, nr)
                            max_r_comp = max(max_r_comp, nr)
                            min_c_comp = min(min_c_comp, nc)
                            max_c_comp = max(max_c_comp, nc)
                components.append((comp_size, min_r_comp, max_r_comp, min_c_comp, max_c_comp))

    if not components:
        return []

    # Select the largest component
    largest = max(components, key=lambda x: x[0])
    _, min_r, max_r, min_c, max_c = largest

    # Extract the subgrid
    subgrid = grid[min_r:max_r + 1, min_c:max_c + 1]
    return subgrid.tolist()