def transform(grid):
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    # Precompute if each column has at least one 8
    has_purple = [False] * cols
    for j in range(cols):
        for i in range(rows):
            if grid[i][j] == 8:
                has_purple[j] = True
                break
    # Initialize output: all positions that were 8 become 0, others 0 (since only 0 and 8)
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    # Process each row
    for i in range(rows):
        j = 0
        while j < cols:
            if grid[i][j] != 0:
                j += 1
                continue
            # Start of run of 0s
            l = j
            while j < cols and grid[i][j] == 0:
                j += 1
            r = j - 1
            # Check if the run is adjacent to at least one 8 in this row
            adjacent_to_8 = False
            if l > 0 and grid[i][l - 1] == 8:
                adjacent_to_8 = True
            if r < cols - 1 and grid[i][r + 1] == 8:
                adjacent_to_8 = True
            if adjacent_to_8:
                for k in range(l, r + 1):
                    if has_purple[k]:
                        output[i][k] = 2
    return output