def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    h = len(grid)
    w = len(grid[0])
    output = [[0] * w for _ in range(h)]
    for i in range(h - 1):
        for j in range(w - 1):
            cells = [
                grid[i][j],
                grid[i][j + 1],
                grid[i + 1][j],
                grid[i + 1][j + 1]
            ]
            count = {}
            for c in cells:
                count[c] = count.get(c, 0) + 1
            for c, cnt in count.items():
                if cnt >= 3 and c != 0:
                    output[i][j] = c
                    output[i][j + 1] = c
                    output[i + 1][j] = c
                    output[i + 1][j + 1] = c
                    break  # Only one such c possible
    return output