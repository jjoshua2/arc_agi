def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0])
    # Find centers of pluses
    centers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                has_up = r > 0 and grid[r-1][c] == 3
                has_down = r < rows - 1 and grid[r+1][c] == 3
                has_left = c > 0 and grid[r][c-1] == 3
                has_right = c < cols - 1 and grid[r][c+1] == 3
                if has_up and has_down and has_left and has_right:
                    centers.append((r, c))
    # Copy grid
    output = [row[:] for row in grid]
    # Process pairs
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            r1, c1 = centers[i]
            r2, c2 = centers[j]
            dr = r2 - r1
            dc = c2 - c1
            adr = abs(dr)
            adc = abs(dc)
            if adr == adc and adr > 0:
                step_r = 1 if dr > 0 else -1
                step_c = 1 if dc > 0 else -1
                for k in range(1, adr):
                    rr = r1 + k * step_r
                    cc = c1 + k * step_c
                    if 0 <= rr < rows and 0 <= cc < cols:
                        output[rr][cc] = 2
    return output