def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    visited = [[False for _ in range(w)] for _ in range(h)]

    def dfs(i: int, j: int) -> None:
        if i < 0 or i >= h or j < 0 or j >= w or visited[i][j] or grid[i][j] != 0:
            return
        visited[i][j] = True
        dfs(i - 1, j)
        dfs(i + 1, j)
        dfs(i, j - 1)
        dfs(i, j + 1)

    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not visited[i][j]:
                count += 1
                dfs(i, j)
    return [[0] for _ in range(count)]