import collections

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    
    # Directions for 4-connected
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Flood fill to mark outside regions (non-5 cells reachable from border)
    outside = [[False] * cols for _ in range(rows)]
    q = collections.deque()
    # Add border non-5 cells
    for r in range(rows):
        for c in [0, cols - 1]:
            if grid[r][c] != 5:
                outside[r][c] = True
                q.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if grid[r][c] != 5 and not outside[r][c]:
                outside[r][c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not outside[nr][nc] and grid[nr][nc] != 5:
                outside[nr][nc] = True
                q.append((nr, nc))
    
    # Visited for components
    visited = [[False] * cols for _ in range(rows)]
    
    # Find templates: enclosed components with color > 1
    templates = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] > 1 and not visited[r][c] and not outside[r][c]:
                # BFS for component
                comp_color = grid[r][c]
                comp = []
                q_comp = collections.deque([(r, c)])
                visited[r][c] = True
                valid = True
                while q_comp and valid:
                    cr, cc = q_comp.popleft()
                    if grid[cr][cc] != comp_color:
                        valid = False
                        break
                    comp.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == comp_color and not outside[nr][nc]:
                            visited[nr][nc] = True
                            q_comp.append((nr, nc))
                if valid and comp:
                    # All enclosed by construction
                    min_r = min(rr for rr, _ in comp)
                    min_c = min(cc for _, cc in comp)
                    shape = frozenset((rr - min_r, cc - min_c) for rr, cc in comp)
                    templates[shape] = comp_color
    
    # Output grid
    output = [row[:] for row in grid]
    
    # Find blue components and recolor if match
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                # BFS for blue component
                comp = []
                q_comp = collections.deque([(r, c)])
                visited[r][c] = True
                while q_comp:
                    cr, cc = q_comp.popleft()
                    comp.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 1:
                            visited[nr][nc] = True
                            q_comp.append((nr, nc))
                if comp:
                    min_r = min(rr for rr, _ in comp)
                    min_c = min(cc for _, cc in comp)
                    shape = frozenset((rr - min_r, cc - min_c) for rr, cc in comp)
                    if shape in templates:
                        new_color = templates[shape]
                        for rr, cc in comp:
                            output[rr][cc] = new_color
    
    return output