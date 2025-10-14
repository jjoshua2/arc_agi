import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape

    def get_components():
        visited = np.zeros((rows, cols), dtype=bool)
        components = []
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == 4 and not visited[i, j]:
                    comp = []
                    stack = [(i, j)]
                    visited[i, j] = True
                    while stack:
                        x, y = stack.pop()
                        comp.append((x, y))
                        for dx, dy in dirs:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 4 and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))
                    components.append(comp)
        return components

    components = get_components()
    if not components:
        return grid.tolist()

    # Find source component with attachments
    source_comp = None
    source_attach = []
    for comp in components:
        attach = []
        comp_set = set(comp)
        seen = set()
        for r, c in comp:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] in (1, 3) and (nr, nc) not in comp_set and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    attach.append(((nr, nc), grid[nr, nc]))
        if attach:
            source_comp = comp
            source_attach = attach
            break

    if not source_comp:
        return grid.tolist()

    output = grid.copy()

    for target_comp in components:
        if target_comp == source_comp:
            continue

        # Compute means
        source_mean_r = np.mean([r for r, c in source_comp])
        source_mean_c = np.mean([c for r, c in source_comp])
        target_mean_r = np.mean([r for r, c in target_comp])
        target_mean_c = np.mean([c for r, c in target_comp])

        best_overlap = -1
        best_a_trans = None

        for flip_v in [False, True]:
            for flip_h in [False, True]:
                for rot in range(4):
                    # Transform source_comp
                    s_trans = []
                    for r, c in source_comp:
                        dx = r - source_mean_r
                        dy = c - source_mean_c
                        if flip_v:
                            dx = -dx
                        if flip_h:
                            dy = -dy
                        dx2 = dx
                        dy2 = dy
                        for _ in range(rot):
                            temp = dx2
                            dx2 = dy2
                            dy2 = -temp
                        nr = target_mean_r + dx2
                        nc = target_mean_c + dy2
                        s_trans.append((round(nr), round(nc)))

                    s_set = set(s_trans)
                    t_set = set(target_comp)
                    overlap = len(s_set.intersection(t_set))

                    # Transform attachments
                    a_trans = []
                    valid = True
                    for (ar, ac), col in source_attach:
                        dx = ar - source_mean_r
                        dy = ac - source_mean_c
                        if flip_v:
                            dx = -dx
                        if flip_h:
                            dy = -dy
                        dx2 = dx
                        dy2 = dy
                        for _ in range(rot):
                            temp = dx2
                            dx2 = dy2
                            dy2 = -temp
                        nr = target_mean_r + dx2
                        nc = target_mean_c + dy2
                        nar = round(nr)
                        nac = round(nc)
                        a_trans.append((nar, nac, col))
                        if not (0 <= nar < rows and 0 <= nac < cols and grid[nar, nac] == 0):
                            valid = False

                    if valid and overlap > best_overlap:
                        best_overlap = overlap
                        best_a_trans = a_trans

        # Place the best
        if best_a_trans:
            for ar, ac, col in best_a_trans:
                output[ar, ac] = col

    return output.tolist()