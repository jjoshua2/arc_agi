def transform(grid: list[list[int]]) -> list[list[int]]:
    # Assume input is 3 rows, 4 columns
    row0 = grid[0]
    row1 = grid[1]
    row2 = grid[2]
    
    # Reverse functions
    def reverse_row(r):
        return r[::-1]
    
    # Compute the three unique rows for output
    out_row0 = reverse_row(row2) + row2
    out_row1 = reverse_row(row1) + row1
    out_row2 = reverse_row(row0) + row0
    
    # Build the 6 rows
    output = [
        out_row0,
        out_row1,
        out_row2,
        out_row2,  # row 3 same as row 2
        out_row1,  # row 4 same as row 1
        out_row0   # row 5 same as row 0
    ]
    
    return output