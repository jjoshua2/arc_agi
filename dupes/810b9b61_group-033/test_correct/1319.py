from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    # First, flood fill exterior 0s
    visited_zero = set()
    q = deque()
    # Add all border 0s
    for r in range(rows):
        # left col
        if grid[r][0] == 0:
            q.append((r, 0))
            visited_zero.add((r, 0))
        # right col
        if grid[r][cols-1] == 0:
            q.append((r, cols-1))
            visited_zero.add((r, cols-1))
    for c in range(1, cols-1):  # top and bottom, avoid corners duplicate
        # top row
        if grid[0][c] == 0:
            q.append((0, c))
            visited_zero.add((0, c))
        # bottom row
        if grid[rows-1][c] == 0:
            q.append((rows-1, c))
            visited_zero.add((rows-1, c))
    # Directions
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited_zero:
                visited_zero.add((nr, nc))
                q.append((nr, nc))
    # Now, enclosed 0s are 0 cells not in visited_zero
    enclosed = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and (r,c) not in visited_zero:
                enclosed.add((r,c))
    # Now, find connected components of 1s
    visited_one = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r,c) not in visited_one:
                # New component
                component = []
                comp_q = deque([(r,c)])
                visited_one.add((r,c))
                while comp_q:
                    cr, cc = comp_q.popleft()
                    component.append((cr,cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1 and (nr,nc) not in visited_one:
                            visited_one.add((nr,nc))
                            comp_q.append((nr,nc))
                # Now, check if this component touches any enclosed
                touches = False
                for pr, pc in component:
                    for dr, dc in dirs:
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) in enclosed:
                            touches = True
                            break
                    if touches:
                        break
                if touches:
                    # Change all in component to 3
                    for pr, pc in component:
                        output[pr][pc] = 3
    return output