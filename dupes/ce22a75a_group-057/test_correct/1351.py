def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = [[0] * cols for _ in range(rows)]
    
    # Find all positions with 5
    fives = [(r, c) for r in range(rows) for c in range(cols) if grid_lst[r][c] == 5]
    
    # For each 5, fill the 3x3 neighborhood with 1s
    for r, c in fives:
        start_row = max(0, r - 1)
        end_row = min(rows, r + 2)
        start_col = max(0, c - 1)
        end_col = min(cols, c + 2)
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                output[i][j] = 1
    
    return output