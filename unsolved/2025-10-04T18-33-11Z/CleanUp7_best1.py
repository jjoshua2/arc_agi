import numpy as np

def transform(grid_lst):
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Find the largest purple (8) rectangle
    purple_color = 8
    purple_mask = grid == purple_color
    
    # Find connected components of purple cells
    visited = np.zeros_like(purple_mask, dtype=bool)
    best_rect = None
    best_area = 0
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(h):
        for c in range(w):
            if purple_mask[r, c] and not visited[r, c]:
                # BFS to find connected purple region
                queue = [(r, c)]
                visited[r, c] = True
                region = []
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                while queue:
                    cr, cc = queue.pop(0)
                    region.append((cr, cc))
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w and 
                            purple_mask[nr, nc] and not visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                # Check if this region is a rectangle and has largest area
                width = max_c - min_c + 1
                height = max_r - min_r + 1
                area = width * height
                
                if area > best_area and len(region) == area:
                    best_area = area
                    best_rect = (min_r, max_r, min_c, max_c)
    
    if best_rect is None:
        return grid.tolist()
    
    min_r, max_r, min_c, max_c = best_rect
    output = grid.copy()
    
    # Project colored cells onto the core boundaries
    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            if color == 0 or color == purple_color:
                continue
                
            # Skip cells inside the core
            if min_r <= r <= max_r and min_c <= c <= max_c:
                continue
                
            # Project to top/bottom if in same column
            if min_c <= c <= max_c:
                if r < min_r:  # Above core
                    output[min_r, c] = color
                elif r > max_r:  # Below core
                    output[max_r, c] = color
                    
            # Project to left/right if in same row
            if min_r <= r <= max_r:
                if c < min_c:  # Left of core
                    output[r, min_c] = color
                elif c > max_c:  # Right of core
                    output[r, max_c] = color
    
    return output.tolist()