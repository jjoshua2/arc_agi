def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows - 2):
        for j in range(cols - 2):
            colors = set()
            all_nonzero = True
            sub = [[0] * 3 for _ in range(3)]
            for di in range(3):
                for dj in range(3):
                    val = grid[i + di][j + dj]
                    sub[di][dj] = val
                    if val == 0:
                        all_nonzero = False
                    else:
                        colors.add(val)
            if all_nonzero and len(colors) == 3:
                return sub
    return []