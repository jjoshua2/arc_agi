import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape
    filled = grid.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Flood fill from borders to mark exterior 0s as -1
    # Left and right borders
    for r in range(rows):
        for c in [0, cols - 1]:
            if filled[r, c] == 0:
                queue = deque([(r, c)])
                filled[r, c] = -1
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and filled[nr, nc] == 0:
                            filled[nr, nc] = -1
                            queue.append((nr, nc))

    # Top and bottom borders
    for c in range(cols):
        for r in [0, rows - 1]:
            if filled[r, c] == 0:
                queue = deque([(r, c)])
                filled[r, c] = -1
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and filled[nr, nc] == 0:
                            filled[nr, nc] = -1
                            queue.append((nr, nc))

    # Create output: original 1s stay 1, exterior 0s stay 0, interior 0s become 2
    output = grid.copy()
    output[filled == 0] = 2

    return output.tolist()