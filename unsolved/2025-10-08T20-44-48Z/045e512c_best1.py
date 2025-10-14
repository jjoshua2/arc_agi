import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    h, w = grid.shape
    output = grid.copy()
    
    # Directions for connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Find template (color 8 connected component)
    visited = np.zeros((h, w), dtype=bool)
    template_min_r = template_max_r = template_min_c = template_max_c = -1
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 8 and not visited[i, j]:
                # Flood fill to find bounding box
                stack = [(i, j)]
                visited[i, j] = True
                min_r, max_r = i, i
                min_c, max_c = j, j
                while stack:
                    x, y = stack.pop()
                    min_r = min(min_r, x)
                    max_r = max(max_r, x)
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 8 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                template_min_r, template_max_r = min_r, max_r
                template_min_c, template_max_c = min_c, max_c
                break
        if template_min_r != -1:
            break
    
    if template_min_r == -1:
        # No template, return original (though additional has one)
        return grid.tolist()
    
    H = template_max_r - template_min_r + 1
    W = template_max_c - template_min_c + 1
    
    # Row masks: H x W binary
    row_masks = []
    for rr in range(H):
        r = template_min_r + rr
        mask = [1 if grid[r, template_min_c + cc] == 8 else 0 for cc in range(W)]
        row_masks.append(mask)
    
    # Column masks: W x H binary
    col_masks = []
    for cc in range(W):
        c = template_min_c + cc
        mask = [1 if grid[template_min_r + rr, c] == 8 else 0 for rr in range(H)]
        col_masks.append(mask)
    
    # Check for vertical seed to the right
    seed_col = template_max_c + 2
    if seed_col < w:
        is_seed = True
        seed_color = grid[template_min_r, seed_col]
        if seed_color == 0 or seed_color == 8:
            is_seed = False
        for rr in range(1, H):
            r = template_min_r + rr
            if grid[r, seed_col] != seed_color:
                is_seed = False
                break
        if is_seed:
            # Grow right using row masks
            current_c = seed_col
            while current_c < w:
                # Place block
                for rr in range(H):
                    r = template_min_r + rr
                    mask = row_masks[rr]
                    for rel_c in range(W):
                        gc = current_c + rel_c
                        if gc >= w:
                            break
                        if mask[rel_c] == 1:
                            output[r, gc] = seed_color
                current_c += W
                if current_c >= w:
                    break
                # Separator
                current_c += 1
                if current_c >= w:
                    break
    
    # Check for horizontal seed below
    seed_row = template_max_r + 2
    if seed_row < h:
        is_seed = True
        seed_color = grid[seed_row, template_min_c]
        if seed_color == 0 or seed_color == 8:
            is_seed = False
        for cc in range(1, W):
            c = template_min_c + cc
            if grid[seed_row, c] != seed_color:
                is_seed = False
                break
        if is_seed:
            # Grow down using col masks
            current_r = seed_row
            while current_r < h:
                # Place block
                for cc in range(W):
                    c = template_min_c + cc
                    mask = col_masks[cc]
                    for rel_r in range(H):
                        gr = current_r + rel_r
                        if gr >= h:
                            break
                        if mask[rel_r] == 1:
                            output[gr, c] = seed_color
                current_r += H
                if current_r >= h:
                    break
                # Separator
                current_r += 1
                if current_r >= h:
                    break
    
    return output.tolist()