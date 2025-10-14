def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    height = len(grid)
    width = len(grid[0])
    non_zeros = []
    sum_r = 0
    sum_c = 0
    count = 0
    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0:
                non_zeros.append((r, c, grid[r][c]))
                sum_r += r
                sum_c += c
                count += 1
    if count == 0:
        return [row[:] for row in grid]
    center_r = round(sum_r / count)
    center_c = round(sum_c / count)
    output = [row[:] for row in grid]
    for r, c, k in non_zeros:
        sym_c = 2 * center_c - c
        sym_r = 2 * center_r - r
        # Horizontal symmetry
        if 0 <= sym_c < width and output[r][sym_c] == 0:
            output[r][sym_c] = k
        # Vertical symmetry
        if 0 <= sym_r < height and output[sym_r][c] == 0:
            output[sym_r][c] = k
        # Both symmetries
        if 0 <= sym_r < height and 0 <= sym_c < width and output[sym_r][sym_c] == 0:
            output[sym_r][sym_c] = k
    return output