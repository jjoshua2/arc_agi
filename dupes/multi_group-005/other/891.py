def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 13 or len(grid[0]) != 6:
        raise ValueError("Input must be 13x6 grid")
    
    # Extract the 3x3 tile from upper part
    tile = [
        [grid[0][0], grid[0][2], grid[0][4]],
        [grid[2][0], grid[2][2], grid[2][4]],
        [grid[4][0], grid[4][2], grid[4][4]]
    ]
    
    # Initialize 18x18 output with zeros
    out = [[0] * 18 for _ in range(18)]
    
    # Process lower part: rows 7 to 12 (indices 7 to 12)
    for k in range(6):
        lower_row = grid[7 + k]
        i = 0
        while i < 6:
            if lower_row[i] != 2:
                i += 1
                continue
            start_col = i
            while i < 6 and lower_row[i] == 2:
                i += 1
            length = i - start_col
            out_row_start = 3 * k
            out_col_base = 3 * start_col
            for tile_idx in range(length):
                out_col_start = out_col_base + 3 * tile_idx
                for tr in range(3):
                    for tc in range(3):
                        r = out_row_start + tr
                        c = out_col_start + tc
                        if 0 <= r < 18 and 0 <= c < 18:
                            out[r][c] = tile[tr][tc]
    
    return out