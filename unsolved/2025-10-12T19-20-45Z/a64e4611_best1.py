from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                # Flood fill to find connected component of 0s
                queue = deque([(i, j)])
                visited[i][j] = True
                component = [(i, j)]
                touches_border = (i == 0 or i == rows - 1 or j == 0 or j == cols - 1)
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            queue.append((nx, ny))
                            component.append((nx, ny))
                            if nx == 0 or nx == rows - 1 or ny == 0 or ny == cols - 1:
                                touches_border = True
                # If the component does not touch the border, it's a hole; fill with 3
                if not touches_border:
                    for x, y in component:
                        output[x][y] = 3
    return output