from collections import deque
import numpy as np

def get_connected_components(grid):
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                minr, maxr, minc, maxc = r, r, c, c
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [ (0,1),(0,-1),(1,0),(-1,0) ]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                            minr = min(minr, nr)
                            maxr = max(maxr, nr)
                            minc = min(minc, nc)
                            maxc = max(maxc, nc)
                components.append({'color': color, 'cells': cells, 'minr': minr, 'maxr': maxr, 'minc': minc, 'maxc': maxc, 'size': len(cells) })
    return components

def transform(grid_lst):
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows, cols = len(grid), len(grid[0])
    components = get_connected_components(grid)
    if not components:
        return grid
    # Find main
    main = max(components, key=lambda x: x['size'])
    main_minr = main['minr']
    main_maxr = main['maxr']
    main_minc = main['minc']
    main_maxc = main['maxc']
    H = main_maxr - main_minr +1
    W = main_maxc - main_minc +1
    unit = H +1 # assume H=W
    mask = [[0 for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[main_minr + i][main_minc + j] != 0:
                mask[i][j] = 1

    # Transpose mask for vertical tiling ? No, for vertical , use the same mask, but the roles swapped? 

    # For now, assume H=W=3 unit=4

    # Remove main from components
    other_components = [comp for comp in components if comp != main]

    for comp in other_components:
        c_minr = comp['minr']
        c_maxr = comp['maxr']
        c_minc = comp['minc']
        c_maxc = comp['maxc']
        C = comp['color']

        # Determine direction
        row_overlap = max(0, min(c_maxr, main_maxr) - max(c_minr, main_minr) +1 )
        col_overlap = max(0, min(c_maxc, main_maxc) - max(c_minc, main_minc) +1 )

        if row_overlap > 0 and col_overlap ==0:
            if c_maxc < main_minc:
                # left horizontal
                col_start = c_maxc - W +1
                for k in range( (main_minc - col_start) // unit +1 ):
                    current_col_start = col_start - k * unit
                    if current_col_start + W -1 < 0:
                        continue
                    for i in range(H):
                        row = main_minr + i
                        for j in range(W):
                            col = current_col_start + j
                            if 0 <= col < cols and mask[i][j] ==1:
                                grid[row][col] = C
            elif c_minc > main_maxc:
                # right horizontal
                col_start = c_minc
                for k in range( (cols - col_start ) // unit +1 ):
                    current_col_start = col_start + k * unit
                    if current_col_start >= cols:
                        continue
                    for i in range(H):
                        row = main_minr + i
                        for j in range(W):
                            col = current_col_start + j
                            if 0 <= col < cols and mask[i][j] ==1:
                                grid[row][col] = C
        if col_overlap > 0 and row_overlap ==0:
            if c_maxr < main_minr:
                # up vertical
                row_start = c_maxr - H +1
                for k in range( (main_minr - row_start ) // unit +1 ):
                    current_row_start = row_start - k * unit
                    if current_row_start + H -1 < 0:
                        continue
                    for i in range(H):
                        row = current_row_start + i
                        for j in range(W):
                            col = main_minc + j
                            if 0 <= row < rows and mask[i][j] ==1:
                                grid[row][col] = C
            elif c_minr > main_maxr:
                # down vertical
                row_start = c_minr
                for k in range( (rows - row_start ) // unit +1 ):
                    current_row_start = row_start + k * unit
                    if current_row_start >= rows:
                        continue
                    for i in range(H):
                        row = current_row_start + i
                        for j in range(W):
                            col = main_minc + j
                            if 0 <= row < rows and mask[i][j] ==1:
                                grid[row][col] = C

    return grid
