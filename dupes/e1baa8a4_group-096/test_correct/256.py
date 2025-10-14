def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    bands = []
    i = 0
    while i < h:
        row = grid[i]
        j = i + 1
        while j < h and grid[j] == row:
            j += 1
        # band from i to j-1, all equal to row
        # find segments in row
        segments = []
        if w > 0:
            curr_color = row[0]
            segments.append(curr_color)
            for k in range(1, w):
                if row[k] != curr_color:
                    curr_color = row[k]
                    segments.append(curr_color)
        bands.append(segments)
        i = j
    # assume all bands have same len(segments)
    num_rows = len(bands)
    if num_rows == 0:
        return []
    num_cols = len(bands[0])
    output = [[0] * num_cols for _ in range(num_rows)]
    for r in range(num_rows):
        output[r] = bands[r][:]
    return output