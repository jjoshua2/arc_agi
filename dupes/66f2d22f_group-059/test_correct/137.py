def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid or not input_grid[0]:
        return []
    rows = len(input_grid)
    cols = len(input_grid[0])
    # Assuming always 4x14 input, 4x7 output
    output = [[0 for _ in range(7)] for _ in range(rows)]
    for r in range(rows):
        for k in range(7):
            left_val = input_grid[r][k]
            right_val = input_grid[r][k + 7]
            if left_val == 0 and right_val == 0:
                output[r][k] = 5
    return output