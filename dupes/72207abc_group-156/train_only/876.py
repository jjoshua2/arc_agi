def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or len(grid_lst) != 3:
        return grid_lst
    middle = grid_lst[1]
    cols = len(middle)
    # Compute triangular positions
    triangular = []
    k = 0
    while True:
        t = k * (k + 1) // 2
        if t >= cols:
            break
        triangular.append(t)
        k += 1
    # Find the length of the initial non-zero block at triangular positions
    l = 0
    for i in range(len(triangular)):
        t = triangular[i]
        if middle[t] == 0:
            break
        l = i + 1
    if l == 0:
        return [row[:] for row in grid_lst]
    # Extract the block
    block = [middle[triangular[i]] for i in range(l)]
    # Create output middle row
    out_middle = middle[:]
    for i in range(len(triangular)):
        t = triangular[i]
        out_middle[t] = block[i % l]
    # Reconstruct output grid
    out = [row[:] for row in grid_lst]
    out[1] = out_middle
    return out