import collections

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [list(row) for row in grid]  # Make a deep copy

    # Find all external zeros (connected to the grid's border via zeros)
    external = set()
    visited = [[False] * cols for _ in range(rows)]
    queue = collections.deque()
    # Add all border zeros to the queue
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows-1 or j == 0 or j == cols-1) and grid[i][j] == 0:
                if not visited[i][j]:
                    queue.append((i, j))
                    visited[i][j] = True
                    external.add((i, j))
    while queue:
        i, j = queue.popleft()
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            ni, nj = i+dx, j+dy
            if 0<=ni<rows and 0<=nj<cols and not visited[ni][nj] and grid[ni][nj] == 0:
                visited[ni][nj] = True
                queue.append((ni, nj))
                external.add((ni, nj))

    # Now, for every cell not in external, if it is 0, set it to 4.
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and (i,j) not in external:
                output[i][j] = 4
    return output