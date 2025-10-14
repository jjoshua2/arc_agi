import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    # Step 1: Keep non-all-zero columns
    keep = [c for c in range(cols) if not np.all(grid[:, c] == 0)]
    new_cols = len(keep)
    new_grid = np.zeros((3, new_cols), dtype=int)
    for i, c in enumerate(keep):
        new_grid[:, i] = grid[:, c]
    # Permanent mask (original)
    permanent = (new_grid != 0) & (new_grid != 5)
    # Step 2: Find groups (connected components of permanent cells)
    visited = np.zeros((3, new_cols), dtype=bool)
    groups = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(3):
        for c in range(new_cols):
            if permanent[r, c] and not visited[r, c]:
                color = new_grid[r, c]
                component = []
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    component.append((rr, cc))
                    for dr, dc in directions:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < 3 and 0 <= nc < new_cols and permanent[nr, nc] and not visited[nr, nc] and new_grid[nr, nc] == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                is_border = any(cc == 0 or cc == new_cols - 1 for _, cc in component)
                groups.append({
                    'color': color,
                    'cells': component,
                    'size': len(component),
                    'border': is_border
                })
    output = new_grid.copy()
    # Step 3: Handle 5's in middle row (row 1)
    for c in range(new_cols):
        if output[1, c] == 5:
            set_color = 0
            left_is_large = False
            right_is_large = False
            right_c = 0
            if c > 0 and permanent[1, c - 1]:
                left_g = next(g for g in groups if any(rr == 1 and cc == c - 1 for rr, cc in g['cells']))
                if left_g['size'] > 1:
                    left_is_large = True
            if c + 1 < new_cols and permanent[1, c + 1]:
                right_g = next(g for g in groups if any(rr == 1 and cc == c + 1 for rr, cc in g['cells']))
                if right_g['size'] > 1:
                    right_is_large = True
                    right_c = right_g['color']
            if right_is_large and not left_is_large:
                set_color = right_c
            output[1, c] = set_color
    # Step 4: Vertical extensions
    # Border groups over 5's vertical
    for g in groups:
        if g['border']:
            for rr, cc in g['cells']:
                if rr > 0 and output[rr - 1, cc] == 5:
                    output[rr - 1, cc] = g['color']
                if rr < 2 and output[rr + 1, cc] == 5:
                    output[rr + 1, cc] = g['color']
    # Internal large down to bottom over 0/5
    for g in groups:
        if not g['border'] and g['size'] > 1:
            for rr, cc in g['cells']:
                output[2, cc] = g['color']
    # Internal singles vertical
    for g in groups:
        if not g['border'] and g['size'] == 1:
            rr, cc = g['cells'][0]
            if rr == 1:  # middle to top
                output[0, cc] = g['color']
            elif rr == 0:  # top to bottom
                output[2, cc] = g['color']
            elif rr == 2:  # bottom to top
                output[0, cc] = g['color']
    # Border bottom large up to middle over 0
    for g in groups:
        if g['border'] and g['size'] > 1 and any(rr == 2 for rr, _ in g['cells']):
            for rr, cc in g['cells']:
                if rr == 2 and output[1, cc] == 0:
                    output[1, cc] = g['color']
    # Step 6: Handle remaining 5's in top and bottom
    # Top row 5's: from left large
    for c in range(new_cols):
        if output[0, c] == 5:
            set_color = 0
            left_is_large = False
            left_c = 0
            right_is_large = False
            if c > 0 and permanent[0, c - 1]:
                left_g = next(g for g in groups if any(rr == 0 and cc_ == c - 1 for rr, cc_ in g['cells']))
                if left_g['size'] > 1:
                    left_is_large = True
                    left_c = left_g['color']
            if c + 1 < new_cols and permanent[0, c + 1]:
                right_g = next(g for g in groups if any(rr == 0 and cc_ == c + 1 for rr, cc_ in g['cells']))
                if right_g['size'] > 1:
                    right_is_large = True
            if left_is_large and not right_is_large:
                set_color = left_c
            output[0, c] = set_color
    # Bottom row 5's: from right large
    for c in range(new_cols):
        if output[2, c] == 5:
            set_color = 0
            left_is_large = False
            right_is_large = False
            right_c = 0
            if c > 0 and permanent[2, c - 1]:
                left_g = next(g for g in groups if any(rr == 2 and cc_ == c - 1 for rr, cc_ in g['cells']))
                if left_g['size'] > 1:
                    left_is_large = True
            if c + 1 < new_cols and permanent[2, c + 1]:
                right_g = next(g for g in groups if any(rr == 2 and cc_ == c + 1 for rr, cc_ in g['cells']))
                if right_g['size'] > 1:
                    right_is_large = True
                    right_c = right_g['color']
            if right_is_large and not left_is_large:
                set_color = right_c
            output[2, c] = set_color
    # Step 7: Additional horizontal over 0's
    # Top: internal middle single extend left 1
    for g in groups:
        if g['size'] == 1 and not g['border'] and any(rr == 1 for rr, _ in g['cells']):
            c_list = [cc for rr, cc in g['cells'] if rr == 1]
            if c_list:
                c = c_list[0]
                if output[0, c] == g['color']:
                    if c > 0 and output[0, c - 1] == 0:
                        output[0, c - 1] = g['color']
    # Bottom: internal top single extend left and right 1
    for g in groups:
        if g['size'] == 1 and not g['border'] and any(rr == 0 for rr, _ in g['cells']):
            c_list = [cc for rr, cc in g['cells'] if rr == 0]
            if c_list:
                c = c_list[0]
                if output[2, c] == g['color']:
                    if c > 0 and output[2, c - 1] == 0:
                        output[2, c - 1] = g['color']
                    if c + 1 < new_cols and output[2, c + 1] == 0:
                        output[2, c + 1] = g['color']
    # Bottom: internal large extend left 1 from min_c
    for g in groups:
        if not g['border'] and g['size'] > 1:
            set_cols = [cc for cc in range(new_cols) if output[2, cc] == g['color']]
            if set_cols:
                min_c = min(set_cols)
                if min_c > 0 and output[2, min_c - 1] == 0:
                    output[2, min_c - 1] = g['color']
    # Middle: border bottom large extend left 1 after up
    for g in groups:
        if g['border'] and g['size'] > 1 and any(rr == 2 for rr, _ in g['cells']):
            new_set_cols = [cc for cc in range(new_cols) if output[1, cc] == g['color'] and not permanent[1, cc]]
            if new_set_cols:
                min_c = min(new_set_cols)
                if min_c > 0 and output[1, min_c - 1] == 0:
                    output[1, min_c - 1] = g['color']
    # Finally, set any remaining 5's to 0 everywhere
    output[output == 5] = 0
    return output.tolist()