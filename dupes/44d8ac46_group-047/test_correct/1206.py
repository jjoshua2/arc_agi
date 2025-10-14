def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    for n in range(1, min(h, w) + 1):
        for r in range(h - n + 1):
            for c in range(w - n + 1):
                # Only consider internal squares
                if r == 0 or r + n == h or c == 0 or c + n == w:
                    continue
                # Check all inside are 0
                all_zero = True
                for i in range(r, r + n):
                    for j in range(c, c + n):
                        if grid[i][j] != 0:
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if not all_zero:
                    continue
                # Check boundaries all 5
                boundaries_ok = True
                # top
                for j in range(c, c + n):
                    if grid[r - 1][j] != 5:
                        boundaries_ok = False
                        break
                if not boundaries_ok:
                    continue
                # bottom
                for j in range(c, c + n):
                    if grid[r + n][j] != 5:
                        boundaries_ok = False
                        break
                if not boundaries_ok:
                    continue
                # left
                for i in range(r, r + n):
                    if grid[i][c - 1] != 5:
                        boundaries_ok = False
                        break
                if not boundaries_ok:
                    continue
                # right
                for i in range(r, r + n):
                    if grid[i][c + n] != 5:
                        boundaries_ok = False
                        break
                if boundaries_ok:
                    # fill with 2
                    for i in range(r, r + n):
                        for j in range(c, c + n):
                            output[i][j] = 2
    return output