import numpy as np
from collections import defaultdict, deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Step 1: Find connected components of 0s (4-connected)
    visited = np.zeros((h, w), dtype=bool)
    comp_id = np.full((h, w), -1, dtype=int)
    components = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                comp_id[r, c] = cid
                comp_cells = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            comp_id[nr, nc] = cid
                            comp_cells.append((nr, nc))
                            stack.append((nr, nc))
                components.append(comp_cells)
                cid += 1
    
    num_comps = len(components)
    if num_comps == 0:
        return grid.tolist()
    
    comp_size = [len(comp) for comp in components]
    
    # Step 2: Build dual graph
    graph = defaultdict(set)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                neigh_comps = set()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                        neigh_comps.add(comp_id[nr, nc])
                neigh_list = list(neigh_comps)
                for i in range(len(neigh_list)):
                    for j in range(i + 1, len(neigh_list)):
                        id1 = neigh_list[i]
                        id2 = neigh_list[j]
                        graph[id1].add(id2)
                        graph[id2].add(id1)
    
    # Step 3: Color the dual graph components, maximizing 4s
    region_color = [-1] * num_comps
    for start_cid in range(num_comps):
        if region_color[start_cid] != -1:
            continue
        # BFS to find one bipartition of this dual cc
        part0 = []  # temp color 0
        part1 = []  # temp color 1
        q = deque([start_cid])
        region_color[start_cid] = 0
        part0.append(start_cid)
        while q:
            u = q.popleft()
            for v in graph[u]:
                if region_color[v] == -1:
                    region_color[v] = 1 - region_color[u]
                    if region_color[v] == 0:
                        part0.append(v)
                    else:
                        part1.append(v)
                    q.append(v)
                # Assume bipartite, no conflict check
        
        # Sizes
        s0 = sum(comp_size[i] for i in part0)
        s1 = sum(comp_size[i] for i in part1)
        
        # Assign 4 to larger, 3 to smaller
        if s0 >= s1:
            for i in part0:
                region_color[i] = 4
            for i in part1:
                region_color[i] = 3
        else:
            for i in part0:
                region_color[i] = 3
            for i in part1:
                region_color[i] = 4
    
    # Step 4: Apply colors to output grid
    out_grid = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                out_grid[r, c] = region_color[comp_id[r, c]]
    
    return out_grid.tolist()