def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    # Find the position of 8
    r8 = c8 = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                r8, c8 = r, c
                break
        if r8 != -1:
            break
    if r8 == -1:
        return output
    # Function to fill zigzag
    def fill_zig(r_start, c_start, dir_r, bar_d):
        curr_r = r_start
        curr_c = c_start
        while 0 <= curr_r < rows:
            output[curr_r][curr_c] = 5
            bar_r = curr_r + dir_r
            if not (0 <= bar_r < rows):
                break
            # Bar columns
            bar_cols = [curr_c + i * bar_d for i in range(3)]
            all_in = all(0 <= c < cols for c in bar_cols)
            if all_in:
                for c in bar_cols:
                    output[bar_r][c] = 5
                # Update for next connect
                next_c = curr_c + 2 * bar_d
                next_r = bar_r + dir_r
                curr_r = next_r
                curr_c = next_c
            else:
                # Degenerate bar at curr_c
                output[bar_r][curr_c] = 5
                break  # Stop after degenerate
    # Upper: up-right
    fill_zig(r8 - 1, c8, -1, 1)
    # Lower: down-left
    fill_zig(r8 + 1, c8, 1, -1)
    return output