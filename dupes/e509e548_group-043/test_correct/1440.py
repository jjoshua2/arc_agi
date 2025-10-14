from collections import deque

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    def get_components():
        visited = [[False] * cols for _ in range(rows)]
        components = []
        for i in range(rows):
            for j in range(cols):
                if grid_lst[i][j] == 3 and not visited[i][j]:
                    component = []
                    q = deque([(i, j)])
                    visited[i][j] = True
                    while q:
                        r, c = q.popleft()
                        component.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid_lst[nr][nc] == 3 and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    components.append(component)
        return components
    
    def get_adj(component):
        adj = {cell: [] for cell in component}
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r, c in component:
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid_lst[nr][nc] == 3 and (nr, nc) in adj:
                    adj[(r, c)].append((nr, nc))
        return adj
    
    components = get_components()
    for comp in components:
        n = len(comp)
        if n == 0:
            continue
        if n == 1:
            r, c = comp[0]
            output[r][c] = 1
            continue
        
        adj = get_adj(comp)
        degrees = {cell: len(adj[cell]) for cell in comp}
        num_ends = sum(1 for d in degrees.values() if d == 1)
        has_branch = any(d > 2 for d in degrees.values())
        
        if num_ends != 2 or has_branch:
            color = 2
        else:
            # Traverse path to count bends
            start = next(cell for cell in comp if degrees[cell] == 1)
            current = start
            prev = None
            directions = []
            while True:
                next_candidates = [nb for nb in adj[current] if nb != prev]
                if not next_candidates:
                    break
                next_cell = next_candidates[0]  # unique for path
                dr = next_cell[0] - current[0]
                dc = next_cell[1] - current[1]
                if dr == -1:
                    dir_ = 'U'
                elif dr == 1:
                    dir_ = 'D'
                elif dc == -1:
                    dir_ = 'L'
                else:  # dc == 1
                    dir_ = 'R'
                directions.append(dir_)
                prev = current
                current = next_cell
            
            bends = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i - 1])
            color = 1 if bends <= 1 else 6
        
        for r, c in comp:
            output[r][c] = color
    
    return output