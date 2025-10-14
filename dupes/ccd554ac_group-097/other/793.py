def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    n = len(grid)
    out_size = n * n
    output = [[0 for _ in range(out_size)] for _ in range(out_size)]
    for tile_i in range(n):
        for tile_j in range(n):
            for r in range(n):
                for c in range(n):
                    output[tile_i * n + r][tile_j * n + c] = grid[r][c]
    return output