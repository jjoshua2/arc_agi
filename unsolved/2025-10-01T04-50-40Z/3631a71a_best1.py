from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    if rows == 0:
        return []
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    visited = set()
    queue = deque()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Enqueue all source cells (1-7)
    for i in range(rows):
        for j in range(cols):
            val = grid_lst[i][j]
            if 1 <= val <= 7:
                queue.append((i, j, val))
                visited.add((i, j))
    # BFS to fill 8 and 9 cells
    while queue:
        r, c, color = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                val = grid_lst[nr][nc]
                if val in (8, 9):
                    visited.add((nr, nc))
                    output[nr][nc] = color
                    queue.append((nr, nc, color))
    return output