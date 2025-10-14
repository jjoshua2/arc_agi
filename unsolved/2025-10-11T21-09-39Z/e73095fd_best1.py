def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Find connected components of 0's using 4-connectivity
    visited = [[False] * cols for _ in range(rows)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                # BFS to find component
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

                # Check if touches border
                touches_border = False
                for pr, pc in component:
                    if pr == 0 or pr == rows - 1 or pc == 0 or pc == cols - 1:
                        touches_border = True
                        break

                # Decide whether to fill
                fill = False
                if not touches_border:
                    fill = True
                elif len(component) <= 4:
                    fill = True

                if fill:
                    for pr, pc in component:
                        output[pr][pc] = 4

    return output