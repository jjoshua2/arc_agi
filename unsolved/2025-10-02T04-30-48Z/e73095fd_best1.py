from collections import deque
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return copy.deepcopy(grid)
    h = len(grid)
    w = len(grid[0])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # First pass: find the maximum component size
    visited = [[False] * w for _ in range(h)]
    max_size = 0
    
    def get_component_size(start_i, start_j):
        size = 0
        q = deque([(start_i, start_j)])
        visited[start_i][start_j] = True
        while q:
            x, y = q.popleft()
            size += 1
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == 0:
                    visited[nx][ny] = True
                    q.append((nx, ny))
        return size
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not visited[i][j]:
                size = get_component_size(i, j)
                if size > max_size:
                    max_size = size
    
    if max_size == 0:
        return copy.deepcopy(grid)
    
    # Second pass: fill small components
    output = copy.deepcopy(grid)
    visited = [[False] * w for _ in range(h)]
    
    def fill_if_small(start_i, start_j):
        component = []
        q = deque([(start_i, start_j)])
        visited[start_i][start_j] = True
        component.append((start_i, start_j))
        size = 1
        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and output[nx][ny] == 0:
                    visited[nx][ny] = True
                    q.append((nx, ny))
                    component.append((nx, ny))
                    size += 1
        if size < max_size:
            for px, py in component:
                output[px][py] = 4
        return size  # not used but for consistency
    
    for i in range(h):
        for j in range(w):
            if output[i][j] == 0 and not visited[i][j]:
                fill_if_small(i, j)
    
    return output