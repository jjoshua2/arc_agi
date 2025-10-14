import numpy as np

def find_components(arr):
    rows, cols = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    components = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] == 9 and not visited[r, c]:
                color = arr[r, c]
                stack = [(r, c)]
                component = []
                visited[r, c] = True
                
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            not visited[nr, nc] and arr[nr, nc] == color):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                
                if component:
                    min_r = min(p[0] for p in component)
                    max_r = max(p[0] for p in component)
                    min_c = min(p[1] for p in component)
                    max_c = max(p[1] for p in component)
                    components.append({
                        'min_r': min_r,
                        'max_r': max_r,
                        'min_c': min_c,
                        'max_c': max_c
                    })
    return components

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst, dtype=int)
    rows, n_cols = grid.shape
    components = find_components(grid)
    if not components:
        return grid.tolist()
    components.sort(key=lambda x: x['min_r'])
    
    # Add local green borders
    for comp in components:
        min_r = comp['min_r']
        max_r = comp['max_r']
        min_c = comp['min_c']
        max_c = comp['max_c']
        # top
        if min_r > 0:
            tr = min_r - 1
            l = max(0, min_c - 1)
            rg = min(n_cols - 1, max_c + 1)
            grid[tr, l:rg + 1] = np.where(grid[tr, l:rg + 1] == 0, 3, grid[tr, l:rg + 1])
        # bottom
        if max_r < rows - 1:
            br = max_r + 1
            l = max(0, min_c - 1)
            rg = min(n_cols - 1, max_c + 1)
            grid[br, l:rg + 1] = np.where(grid[br, l:rg + 1] == 0, 3, grid[br, l:rg + 1])
        # left
        l = max(0, min_c - 1)
        for rr in range(min_r, max_r + 1):
            if grid[rr, l] == 0:
                grid[rr, l] = 3
        # right
        rg = min(n_cols - 1, max_c + 1)
        for rr in range(min_r, max_r + 1):
            if grid[rr, rg] == 0:
                grid[rr, rg] = 3
    
    # Horizontal connections for close pairs
    for i in range(len(components) - 1):
        c1 = components[i]
        c2 = components[i + 1]
        if c1['max_r'] + 2 >= c2['min_r']:
            upper_bottom = c1['max_r'] + 1
            if upper_bottom >= c2['min_r']:
                # include lower
                conn_start = c2['min_r'] - 1
                conn_end = c2['max_r']
                left = max(0, c2['min_c'] - 1)
                right = min(n_cols - 1, c1['max_c'] + 1)
                for rr in range(max(0, conn_start), min(rows, conn_end + 1)):
                    mask = (grid[rr, left:right + 1] == 0) | (grid[rr, left:right + 1] == 1)
                    grid[rr, left:right + 1][mask] = 3
    
    # Blue corridors for non-last
    if components:
        overall_max_r = components[-1]['max_r']
        overall_end = overall_max_r + 1
        for i in range(len(components) - 1):
            c = components[i]
            start_r = c['max_r'] + 2
            minc, maxc = c['min_c'], c['max_c']
            for rr in range(max(0, start_r), min(rows, overall_end + 1)):
                for cc in range(minc, maxc + 1):
                    if grid[rr, cc] == 0:
                        grid[rr, cc] = 1
    
    # Extra bottom for all
    if components:
        overall_bottom = components[-1]['max_r'] + 1
        extra_start = overall_bottom + 1
        extra_end = min(rows - 1, overall_bottom + 3)
        if extra_start <= extra_end:
            for comp in components:
                minc, maxc = comp['min_c'], comp['max_c']
                for rr in range(extra_start, extra_end + 1):
                    for cc in range(max(0, minc), min(n_cols, maxc + 1)):
                        if grid[rr, cc] == 0:
                            grid[rr, cc] = 1
    
    # Last component bottom extend down 1 row, override to 3
    if components:
        last = components[-1]
        bottom_r = last['max_r'] + 1
        if bottom_r < rows - 1:
            ext_r = bottom_r + 1
            l = max(0, last['min_c'] - 1)
            rg = min(n_cols - 1, last['max_c'] + 1)
            for cc in range(l, rg + 1):
                grid[ext_r, cc] = 3  # override
    
    # Last component extra from +2, set 1 if 0
    if components:
        last = components[-1]
        overall_bottom = last['max_r'] + 1
        extra_start_last = overall_bottom + 2
        extra_end = min(rows - 1, overall_bottom + 3)
        if extra_start_last <= extra_end:
            minc, maxc = last['min_c'], last['max_c']
            for rr in range(extra_start_last, extra_end + 1):
                for cc in range(max(0, minc), min(n_cols, maxc + 1)):
                    if grid[rr, cc] == 0:
                        grid[rr, cc] = 1
    
    return grid.tolist()