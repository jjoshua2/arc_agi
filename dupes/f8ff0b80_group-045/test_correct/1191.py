import numpy as np

def transform(grid_lst):
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]  # 8 directions
    for i in range(rows):
        for j in range(cols):
            if grid[i,j] != 0 and not visited[i,j]:
                color = grid[i,j]
                size = 0
                stack = [(i,j)]
                visited[i,j] = True
                while stack:
                    x, y = stack.pop()
                    size += 1
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx,ny] and grid[nx,ny] == color:
                            visited[nx,ny] = True
                            stack.append((nx,ny))
                components.append((size, color))
    # Sort by size descending
    components.sort(key=lambda x: -x[0])
    # Take top 3
    colors = [comp[1] for comp in components[:3]]
    return [[c] for c in colors]