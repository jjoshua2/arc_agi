def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                continue
            filled = False
            for dr in [0, -1]:
                for dc in [0, -1]:
                    tr = r + dr
                    tc = c + dc
                    if tr < 0 or tc < 0 or tr + 1 >= rows or tc + 1 >= cols:
                        continue
                    all_zero = True
                    for i in range(2):
                        for j in range(2):
                            if grid[tr + i][tc + j] != 0:
                                all_zero = False
                                break
                        if not all_zero:
                            break
                    if all_zero:
                        filled = True
                        break
                if filled:
                    break
            if filled:
                result[r][c] = 2
    return result