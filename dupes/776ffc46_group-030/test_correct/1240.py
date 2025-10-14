import collections

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows, cols = len(grid), len(grid[0])
    
    # Mark outside: non-5 cells reachable from border
    outside = [[False] * cols for _ in range(rows)]
    q = collections.deque()
    # Add border non-5 cells
    for r in range(rows):
        for c in [0, cols - 1]:
            if grid[r][c] != 5 and not outside[r][c]:
                outside[r][c] = True
                q.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if grid[r][c] != 5 and not outside[r][c]:
                outside[r][c] = True
                q.append((r, c))
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
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
                component = []
                q_comp = collections.deque([(r, c)])
                visited[r][c] = True
                while q_comp:
                    cr, cc = q_comp.popleft()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]
                            and grid[nr][nc] == comp_color and not outside[nr][nc]):
                            visited[nr][nc] = True
                            q_comp.append((nr, nc))
                if component:
                    # All enclosed by construction
                    min_r = min(rr for rr, _ in component)
                    min_c = min(cc for _, cc in component)
                    shape = frozenset((rr - min_r, cc - min_c) for rr, cc in component)
                    templates[shape] = comp_color
    
    # Output grid
    output = [row[:] for row in grid]
    
    # Find blue components and recolor if match
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                # BFS for blue component
                component = []
                q_comp = collections.deque([(r, c)])
                visited[r][c] = True
                while q_comp:
                    cr, cc = q_comp.popleft()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 1:
                            visited[nr][nc] = True
                            q_comp.append((nr, nc))
                if component:
                    min_r = min(rr for rr, _ in component)
                    min_c = min(cc for _, cc in component)
                    shape = frozenset((rr - min_r, cc - min_c) for rr, cc in component)
                    if shape in templates:
                        new_color = templates[shape]
                        for rr, cc in component:
                            output[rr][cc] = new_color
    
    return output