def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])

    def scan_line(line):
        n = len(line)
        i = 0
        current_length = 0
        seen = False
        while i < n:
            if line[i] == 2:
                current_length += 1
                seen = True
                i += 1
                continue
            if line[i] == 0 and seen:
                line[i] = 2
                current_length += 1
                i += 1
                continue
            if line[i] == 5:
                if current_length >= 2:
                    seen = True
                else:
                    seen = False
                current_length = 0
                i += 1
                continue
            seen = False
            current_length = 0
            i += 1
        return line

    changed = True
    while changed:
        changed = False
        # Horizontal
        for r in range(rows):
            old = grid[r][:]
            scan_line(grid[r])
            if grid[r] != old:
                changed = True
        # Vertical
        for c in range(cols):
            old_line = [grid[r][c] for r in range(rows)]
            line = [grid[r][c] for r in range(rows)]
            scan_line(line)
            if line != old_line:
                changed = True
                for r in range(rows):
                    grid[r][c] = line[r]

    return grid