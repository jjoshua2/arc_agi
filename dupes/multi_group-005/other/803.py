def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 10 or len(grid[0]) != 21:
        raise ValueError("Input must be 10x21 grid")
    h = 10
    w_out = 10
    out = [[0 for _ in range(w_out)] for _ in range(h)]
    for r in range(h):
        for c in range(w_out):
            left = grid[r][c]
            right = grid[r][c + 11]
            if left != 0 and right != 0:
                out[r][c] = left
            elif left != 0:
                out[r][c] = 2
            elif right != 0:
                out[r][c] = 1
            else:
                out[r][c] = 0
    return out