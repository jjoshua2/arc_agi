def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(rows - 2):
        for j in range(cols - 2):
            # Collect non-zero cells in this 3x3
            component_pos = []
            for di in range(3):
                for dj in range(3):
                    r = i + di
                    c = j + dj
                    if grid[r][c] != 0:
                        component_pos.append((r, c))
            if not component_pos:
                continue

            # Check if they form a single connected component and span the full 3x3
            start = component_pos[0]
            visited = set()
            stack = [start]
            visited.add(start)
            while stack:
                cr, cc = stack.pop()
                for dr, dc in directions:
                    nr = cr + dr
                    nc = cc + dc
                    if (nr, nc) in component_pos and (nr, nc) not in visited and i <= nr <= i + 2 and j <= nc <= j + 2:
                        visited.add((nr, nc))
                        stack.append((nr, nc))

            if len(visited) != len(component_pos):
                continue  # Not fully connected

            # Check bounding box spans exactly the 3x3
            min_r_comp = min(r for r, c in visited)
            max_r_comp = max(r for r, c in visited)
            min_c_comp = min(c for r, c in visited)
            max_c_comp = max(c for r, c in visited)
            if min_r_comp != i or max_r_comp != i + 2 or min_c_comp != j or max_c_comp != j + 2:
                continue

            # Trigger fill the outer 5x5 perimeter
            top_r = i - 1
            bot_r = i + 3
            left_c_fill = j - 1
            right_c_fill = j + 3

            # Fill top
            if top_r >= 0:
                for cc in range(max(0, left_c_fill), min(cols, right_c_fill + 1)):
                    out[top_r][cc] = 5

            # Fill bottom
            if bot_r < rows:
                for cc in range(max(0, left_c_fill), min(cols, right_c_fill + 1)):
                    out[bot_r][cc] = 5

            # Fill left
            if left_c_fill >= 0:
                for rr in range(max(0, top_r), min(rows, bot_r + 1)):
                    out[rr][left_c_fill] = 5

            # Fill right
            if right_c_fill < cols:
                for rr in range(max(0, top_r), min(rows, bot_r + 1)):
                    out[rr][right_c_fill] = 5

    return out