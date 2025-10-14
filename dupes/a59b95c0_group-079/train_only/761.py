def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    # Assume 3x3 input
    unique_colors = set()
    for row in grid:
        unique_colors.update(row)
    K = len(unique_colors)
    
    # Extended rows: each original row repeated K times horizontally
    extended_rows = []
    for orig_row in grid:
        ext_row = orig_row * K  # concatenate K times
        extended_rows.append(ext_row)
    
    # Output: the block of 3 extended rows, repeated K times vertically
    output = extended_rows * K
    
    return output