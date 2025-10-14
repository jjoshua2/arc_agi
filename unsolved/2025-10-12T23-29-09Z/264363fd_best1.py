import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape

    # Find background color
    unique, counts = np.unique(grid, return_counts=True)
    background = unique[np.argmax(counts)]

    # Find connected components of non-background cells
    visited = np.zeros((rows, cols), bool)
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != background and not visited[i, j]:
                component = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y, grid[x, y]))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != background and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                components.append(component)

    # Classify mains and patterns
    THRESHOLD = 50
    mains = [comp for comp in components if len(comp) > THRESHOLD]
    patterns = [comp for comp in components if len(comp) <= THRESHOLD]

    # Assume one pattern
    pattern = patterns[0]

    # Compute centroid
    num = len(pattern)
    sum_r = sum(r for r, c, col in pattern)
    sum_c = sum(c for r, c, col in pattern)
    center_r = round(sum_r / num)
    center_c = round(sum_c / num)

    # Find center_color
    center_color = next(col for r, c, col in pattern if r == center_r and c == center_c)

    # Create output
    output = grid.copy()

    # Remove patterns
    for pat in patterns:
        for r, c, col in pat:
            output[r, c] = background

    # Process each main
    for main in mains:
        main_set = set((r, c) for r, c, col in main)
        # Find seeds
        seeds = [(r, c) for r, c, col in main if col == center_color]

        # For each seed
        for sr, sc in seeds:
            # Place pattern
            for pr, pc, pcol in pattern:
                dr = pr - center_r
                dc = pc - center_c
                nr = sr + dr
                nc = sc + dc
                if (nr, nc) in main_set:
                    output[nr, nc] = pcol

            # Extend horizontal center row
            horiz_dcs = set(dc for pr, pc, pcol in pattern if pr == center_r for dc in [pc - center_c])
            if horiz_dcs:
                min_dc = min(horiz_dcs)
                left_color = next(pcol for pr, pc, pcol in pattern if pr == center_r and pc - center_c == min_dc)
                max_dc = max(horiz_dcs)
                right_color = next(pcol for pr, pc, pcol in pattern if pr == center_r and pc - center_c == max_dc)

                # Extend left
                nc = sc + min_dc - 1
                while nc >= 0 and (sr, nc) in main_set:
                    output[sr, nc] = left_color
                    nc -= 1

                # Extend right
                nc = sc + max_dc + 1
                while nc < cols and (sr, nc) in main_set:
                    output[sr, nc] = right_color
                    nc += 1

            # Extend vertical center column
            vert_drs = set(dr for pr, pc, pcol in pattern if pc == center_c for dr in [pr - center_r])
            if vert_drs:
                min_dr = min(vert_drs)
                up_color = next(pcol for pr, pc, pcol in pattern if pc == center_c and pr - center_r == min_dr)
                max_dr = max(vert_drs)
                down_color = next(pcol for pr, pc, pcol in pattern if pc == center_c and pr - center_r == max_dr)

                # Extend up
                nr = sr + min_dr - 1
                while nr >= 0 and (nr, sc) in main_set:
                    output[nr, sc] = up_color
                    nr -= 1

                # Extend down
                nr = sr + max_dr + 1
                while nr < rows and (nr, sc) in main_set:
                    output[nr, sc] = down_color
                    nr += 1

    return output.tolist()