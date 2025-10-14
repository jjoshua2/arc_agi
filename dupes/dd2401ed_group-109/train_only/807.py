def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]

    # Find wall_col
    wall_col = -1
    for c in range(w):
        if all(grid[r][c] == 5 for r in range(h)):
            wall_col = c
            break
    if wall_col == -1:
        return output  # No wall, no change

    # Compute n and l
    n = 0
    l = 0
    for c in range(wall_col):
        count1 = sum(1 for r in range(h) if grid[r][c] == 1)
        n += count1
        l = max(l, count1)

    # Compute e and update current (skip initial empty columns)
    e = 0
    current = wall_col
    while current + 1 < w:
        num2 = sum(1 for r in range(h) if grid[r][current + 1] == 2)
        if num2 == 0:
            e += 1
            current += 1
        else:
            break

    # Compute additional
    additional = 0
    broke = False
    while additional < n and current + 1 < w:
        next_c = current + 1
        num2 = sum(1 for r in range(h) if grid[r][next_c] == 2)
        if num2 > l:
            broke = True
            break
        if num2 == 0:
            current = next_c
            continue
        # num2 > 0 and <= l
        additional += 1
        current = next_c

    # Determine new_wall
    if broke:
        new_wall = current
    else:
        new_wall = current + 1

    # Flip 2's to 1's if wall_col > 3
    if wall_col > 3:
        for c in range(wall_col + 1, new_wall):
            for r in range(h):
                if output[r][c] == 2:
                    output[r][c] = 1

    # Clear old wall
    for r in range(h):
        output[r][wall_col] = 0

    # Place new wall
    for r in range(h):
        output[r][new_wall] = 5

    return output