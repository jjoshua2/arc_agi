from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Step 1: Label connected components of 8's
    label_8 = [[-1] * cols for _ in range(rows)]
    cid = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8 and label_8[i][j] == -1:
                # Iterative DFS to label component
                stack = [(i, j)]
                label_8[i][j] = cid
                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 8 and label_8[nx][ny] == -1:
                            label_8[nx][ny] = cid
                            stack.append((nx, ny))
                cid += 1
    num_components = cid
    
    # Step 2: Mark outside 0's using BFS from border 0's
    visited_0 = [[False] * cols for _ in range(rows)]
    q = deque()
    # Add border 0's to queue
    for i in range(rows):
        if grid[i][0] == 0 and not visited_0[i][0]:
            q.append((i, 0))
            visited_0[i][0] = True
        if grid[i][cols - 1] == 0 and not visited_0[i][cols - 1]:
            q.append((i, cols - 1))
            visited_0[i][cols - 1] = True
    for j in range(cols):
        if grid[0][j] == 0 and not visited_0[0][j]:
            q.append((0, j))
            visited_0[0][j] = True
        if grid[rows - 1][j] == 0 and not visited_0[rows - 1][j]:
            q.append((rows - 1, j))
            visited_0[rows - 1][j] = True
    # BFS to mark all reachable 0's as outside
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and not visited_0[nx][ny]:
                visited_0[nx][ny] = True
                q.append((nx, ny))
    
    # Step 3: Find inner 0-components and assign to enclosing C
    enclosed_count = [0] * num_components
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited_0[i][j]:
                # New inner 0-component, collect cells and adjacent CIDs using iterative DFS
                inner_cells = []
                adj_cids = set()
                stack = [(i, j)]
                visited_0[i][j] = True
                inner_cells.append((i, j))
                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if grid[nx][ny] == 0 and not visited_0[nx][ny]:
                                visited_0[nx][ny] = True
                                stack.append((nx, ny))
                                inner_cells.append((nx, ny))
                            elif grid[nx][ny] == 8 and label_8[nx][ny] != -1:
                                adj_cids.add(label_8[nx][ny])
                # If enclosed by exactly one component
                if len(adj_cids) == 1:
                    the_cid = next(iter(adj_cids))
                    enclosed_count[the_cid] += 1
    
    # Step 4: Color the 8's in each component
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8:
                cid = label_8[i][j]
                color = enclosed_count[cid]
                output[i][j] = color
    
    return output