from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    if rows == 0:
        return []
    cols = len(grid_lst[0])
    visited = set()
    queue = deque()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Add border 0's to queue
    for i in range(rows):
        if grid_lst[i][0] == 0:
            queue.append((i, 0))
            visited.add((i, 0))
        if grid_lst[i][cols - 1] == 0:
            queue.append((i, cols - 1))
            visited.add((i, cols - 1))
    for j in range(cols):
        if grid_lst[0][j] == 0:
            queue.append((0, j))
            visited.add((0, j))
        if grid_lst[rows - 1][j] == 0:
            queue.append((rows - 1, j))
            visited.add((rows - 1, j))
    # BFS flood fill
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid_lst[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    # Create output
    output = [row[:] for row in grid_lst]
    for i in range(rows):
        for j in range(cols):
            if grid_lst[i][j] == 0 and (i, j) not in visited:
                output[i][j] = 8
    return output