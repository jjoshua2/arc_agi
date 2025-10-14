def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    assert h == 4 and w == 4, "Input must be 4x4"
    out_h, out_w = 16, 16
    output = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for meta_i in range(4):
        for meta_j in range(4):
            if grid[meta_i][meta_j] == 0:
                start_r = 4 * meta_i
                start_c = 4 * meta_j
                for dr in range(4):
                    for dc in range(4):
                        output[start_r + dr][start_c + dc] = 0
            else:
                start_r = 4 * meta_i
                start_c = 4 * meta_j
                for dr in range(4):
                    for dc in range(4):
                        output[start_r + dr][start_c + dc] = grid[dr][dc]
    return output