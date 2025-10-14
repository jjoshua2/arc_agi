def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    
    # Check if left-leaning: any 2 in column 0
    left_leaning = any(row[0] == 2 for row in grid)
    
    # Compute transformed rows
    trans_rows = []
    for row in grid:
        # Reverse the row
        rev_row = row[::-1]
        # Invert: 0 -> 8, 2 -> 0 (assuming only 0 and 2)
        trans_row = [8 if cell == 0 else 0 for cell in rev_row]
        trans_rows.append(trans_row)
    
    # Build output
    out_rows = []
    if left_leaning:
        for i in range(h):
            out_rows.append(trans_rows[i] + grid[i])
    else:
        for i in range(h):
            out_rows.append(grid[i] + trans_rows[i])
    
    return out_rows