def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    slabs = []
    i = 0
    while i < h:
        row = grid[i]
        start = i
        while i < h and grid[i] == row:
            i += 1
        # extract segments from row
        segments = []
        if row:
            curr = row[0]
            segments.append(curr)
            for j in range(1, len(row)):
                if row[j] != curr:
                    curr = row[j]
                    segments.append(curr)
        slabs.append(segments)
    return slabs