def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            if grid_lst[r][c] != 0 and not visited[r][c]:
                color = grid_lst[r][c]
                component_positions = []
                stack = [(r, c)]
                visited[r][c] = True
                min_r, max_r, min_c, max_c = r, r, c, c
                while stack:
                    x, y = stack.pop()
                    component_positions.append((x, y))
                    min_r = min(min_r, x)
                    max_r = max(max_r, x)
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid_lst[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                if component_positions:  # Only add if there are cells
                    components.append({
                        'positions': component_positions,
                        'bbox': (min_r, max_r, min_c, max_c)
                    })

    # Identify top-level components (those not contained in any other)
    n = len(components)
    is_top_level = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            comp_i = components[i]['bbox']
            comp_j = components[j]['bbox']
            if (comp_j[0] <= comp_i[0] and comp_j[1] >= comp_i[1] and
                comp_j[2] <= comp_i[2] and comp_j[3] >= comp_i[3]):
                is_top_level[i] = False
                break

    # Create output grid
    output_grid = [row[:] for row in grid_lst]

    # Remove top-level components
    for i in range(n):
        if is_top_level[i]:
            for r, c in components[i]['positions']:
                output_grid[r][c] = 0

    return output_grid