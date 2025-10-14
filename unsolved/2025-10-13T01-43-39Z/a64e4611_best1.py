from collections import deque
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    grid = np.array(grid)
    rows, cols = grid.shape
    
    # Find c
    all_colors = set()
    for row in grid:
        for val in row:
            if val != 0:
                all_colors.add(val)
    if len(all_colors) != 1:
        # Assume one non-zero color
        return grid.tolist()
    c = list(all_colors)[0]
    
    # Get components
    def get_components():
        visited = set()
        components = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == c and (i, j) not in visited:
                    comp = []
                    queue = deque([(i, j)])
                    visited.add((i, j))
                    while queue:
                        x, y = queue.popleft()
                        comp.append((x, y))
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == c and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny))
                    components.append(comp)
        return components
    
    components = get_components()
    if not components:
        return grid.tolist()
    
    sizes = [len(comp) for comp in components]
    max_size = max(sizes)
    main_index = sizes.index(max_size)
    main_cells = set(components[main_index])
    
    # Compute w
    small_counts_per_row = [0] * rows
    for ii in range(rows):
        for jj in range(cols):
            if grid[ii, jj] == c and (ii, jj) not in main_cells:
                small_counts_per_row[ii] += 1
    w = (max(small_counts_per_row) + 1) if small_counts_per_row else 0
    
    # min_col
    if not main_cells:
        return grid.tolist()
    min_col = min(j for i, j in main_cells)
    
    bar_start = min_col - w
    bar_end = min_col - 1
    
    # threshold
    threshold = cols // 3
    
    # compute gap_sizes
    gap_sizes = []
    for r in range(rows):
        left_js = [j for j in range(cols) if grid[r, j] == c and (r, j) not in main_cells]
        right_js = [j for j in range(cols) if grid[r, j] == c and (r, j) in main_cells]
        max_l = max(left_js) if left_js else -1
        min_r = min(right_js) if right_js else cols
        g = min_r - max_l - 1 if max_l >= 0 and min_r < cols else (float('inf') if max_l >= 0 else 0)
        gap_sizes.append(g)
    
    # find start_row
    start_row = rows
    for r in range(rows):
        if gap_sizes[r] >= threshold:
            start_row = r
            break
    end_row = rows - 1
    
    # fill
    output = grid.copy()
    for r in range(start_row, end_row + 1):
        # main bar
        for cc in range(max(0, bar_start), min(cols, bar_end + 1)):
            if output[r, cc] == 0:
                output[r, cc] = 3
        # extensions
        small_count = small_counts_per_row[r]
        main_count = sum(1 for j in range(cols) if grid[r, j] == c and (r, j) in main_cells)
        if small_count == 2 and main_count == 0:
            for cc in range(bar_end + 1, cols):
                if output[r, cc] == 0:
                    output[r, cc] = 3
        if small_count == 0 and main_count <= 5:
            for cc in range(0, bar_start):
                if output[r, cc] == 0:
                    output[r, cc] = 3
    
    return output.tolist()