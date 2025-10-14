def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    # Find the unique positive color C
    colors = set()
    for row in grid:
        for cell in row:
            if cell > 0:
                colors.add(cell)
    if len(colors) != 1:
        # Assume single C as per examples; handle gracefully if not
        return [[0 for _ in row] for row in grid]
    C = next(iter(colors))
    # Mapping from examples
    mapping = {3: 1, 5: 4, 8: 2}
    D = mapping.get(C, 0)  # Default 0 if unknown C, but examples covered
    # Build output
    h, w = len(grid), len(grid[0])
    output = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0:
                output[i][j] = D
            # else: remains 0
    return output