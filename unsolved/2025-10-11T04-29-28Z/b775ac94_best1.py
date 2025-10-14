def transform(grid: list[list[int]]) -> list[list[int]]:
    # Implement the junction-driven quadrant expansion described above.
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]  # copy for writing

    # 4-neighbor directions
    neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(i, j):
        return 0 <= i < h and 0 <= j < w

    # BFS flood-fill to get component of non-zero cells starting from given seeds
    def get_component(seeds):
        comp = set()
        stack = list(seeds)
        while stack:
            x, y = stack.pop()
            if not in_bounds(x, y):
                continue
            if (x, y) in comp:
                continue
            if grid[x][y] == 0:
                continue
            comp.add((x, y))
            for dx, dy in neigh4:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny) and (nx, ny) not in comp and grid[nx][ny] != 0:
                    stack.append((nx, ny))
        return comp

    # Find nearest non-zero color in a component to a given coordinate (Manhattan distance)
    def nearest_color_in_comp(comp, x, y):
        best = None
        bestd = None
        for (cx, cy) in comp:
            col = grid[cx][cy]
            if col == 0:
                continue
            d = abs(cx - x) + abs(cy - y)
            if bestd is None or d < bestd or (d == bestd and (col < best)):
                bestd = d
                best = col
        return best if best is not None else 0

    # Iterate all possible 2x2 top-left positions and detect junctions
    processed_rects = []  # keep rectangles we've filled (optional)
    for r in range(h - 1):
        for c in range(w - 1):
            # Gather the four corner colors excluding 0
            corners = [
                grid[r][c],
                grid[r][c + 1],
                grid[r + 1][c],
                grid[r + 1][c + 1]
            ]
            nonzero_colors = set([col for col in corners if col != 0])
            # The examples react when at least 3 distinct non-zero colors are present;
            # use threshold >= 3 (accept also 4)
            if len(nonzero_colors) >= 3:
                # Seeds for the component are the non-zero corner positions
                seeds = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                seeds = [(x, y) for (x, y) in seeds if in_bounds(x, y) and grid[x][y] != 0]
                if not seeds:
                    continue
                comp = get_component(seeds)
                if not comp:
                    continue
                minr = min(x for x, _ in comp)
                maxr = max(x for x, _ in comp)
                minc = min(y for _, y in comp)
                maxc = max(y for _, y in comp)

                # dividing lines: column between c and c+1, row between r-1 and r (if r>0)
                center_col = c
                center_row = r - 1 if r > 0 else r

                # compute symmetric extents so the rectangle includes the component
                L = max(center_col - minc, maxc - (center_col + 1))
                T = max(center_row - minr, maxr - (center_row + 1))

                row_start = max(0, center_row - T)
                row_end = min(h - 1, center_row + 1 + T)
                col_start = max(0, center_col - L)
                col_end = min(w - 1, center_col + 1 + L)

                # Map corner colors; if a corner is zero, choose nearest non-zero in comp
                TL = grid[r][c]
                TR = grid[r][c + 1]
                BL = grid[r + 1][c]
                BR = grid[r + 1][c + 1]

                if TL == 0:
                    TL = nearest_color_in_comp(comp, r, c) or TL
                if TR == 0:
                    TR = nearest_color_in_comp(comp, r, c + 1) or TR
                if BL == 0:
                    BL = nearest_color_in_comp(comp, r + 1, c) or BL
                if BR == 0:
                    BR = nearest_color_in_comp(comp, r + 1, c + 1) or BR

                # Fill the rectangle quadrants without overwriting original non-zero cells
                for i in range(row_start, row_end + 1):
                    for j in range(col_start, col_end + 1):
                        if out[i][j] != 0:
                            # preserve existing colored pixels
                            continue
                        # Determine quadrant relative to dividing lines:
                        # top half: i <= center_row
                        # bottom half: i >= center_row + 1
                        # left half: j <= center_col
                        # right half: j >= center_col + 1
                        if i <= center_row and j <= center_col:
                            color = TL
                        elif i <= center_row and j >= center_col + 1:
                            color = TR
                        elif i >= center_row + 1 and j <= center_col:
                            color = BL
                        else:
                            color = BR
                        # Only fill if color is non-zero
                        if color != 0:
                            out[i][j] = color

                processed_rects.append((row_start, row_end, col_start, col_end))

    return out