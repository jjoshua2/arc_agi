import sys

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    min_r = sys.maxsize
    max_r = -1
    min_c = sys.maxsize
    max_c = -1
    found = False
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                found = True
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    if not found:
        return []
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    output = []
    for i in range(height):
        global_r = min_r + i
        row = []
        for j in range(width):
            global_c = min_c + (width - 1 - j)
            row.append(grid[global_r][global_c])
        output.append(row)
    return output