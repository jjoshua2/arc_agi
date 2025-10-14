def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    if rows == 0:
        return []
    cols = len(grid_lst[0])
    grid = grid_lst  # alias for convenience

    def get_component(start_i, start_j, visited):
        stack = [(start_i, start_j)]
        comp = []
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while stack:
            ci, cj = stack.pop()
            if (ci, cj) in visited:
                continue
            visited.add((ci, cj))
            if not (0 <= ci < rows and 0 <= cj < cols and grid[ci][cj] == 1):
                continue
            comp.append((ci, cj))
            for di, dj in dirs:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == 1 and (ni, nj) not in visited:
                    stack.append((ni, nj))
        return comp

    visited = set()
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and (i, j) not in visited:
                comp = get_component(i, j, visited)
                if comp:
                    min_r = min(x[0] for x in comp)
                    min_c = min(x[1] for x in comp)
                    components.append((min_r, min_c, comp))

    components.sort(key=lambda x: (x[0], x[1]))

    seeds = []
    for i in range(rows):
        for j in range(cols):
            val = grid[i][j]
            if val != 0 and val != 1:
                seeds.append(val)

    output = [row[:] for row in grid_lst]
    for idx, (_, _, comp) in enumerate(components):
        if idx >= len(seeds):
            break  # safety, though should not happen
        color = seeds[idx]
        for ci, cj in comp:
            output[ci][cj] = color

    return output