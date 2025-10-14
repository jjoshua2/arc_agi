import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Flood fill from border 0's to mark outer space
    for i in range(rows):
        for j in range(cols):
            if (grid[i, j] == 0 and not visited[i, j] and
                (i == 0 or i == rows - 1 or j == 0 or j == cols - 1)):
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            queue.append((nx, ny))

    # Create output
    output = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0:
                output[i, j] = grid[i, j]
            elif visited[i, j]:
                output[i, j] = 0
            else:
                output[i, j] = 3
    return output.tolist()