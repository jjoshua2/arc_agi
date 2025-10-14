from collections import deque
import copy

def get_components(grid):
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    pos_to_comp = {}
    comp_id = 0
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                this_comp = []
                q = deque([(i, j)])
                visited[i][j] = True
                color = grid[i][j]
                while q:
                    x, y = q.popleft()
                    this_comp.append((x, y))
                    pos_to_comp[(x, y)] = comp_id
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                components.append(this_comp)
                comp_id += 1
    return components, pos_to_comp

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    components, pos_to_comp = get_components(grid)
    new_grid = copy.deepcopy(grid)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows = len(grid)
    cols = len(grid[0])
    for cid, comp in enumerate(components):
        if len(comp) != 1:
            continue
        i, j = comp[0]
        T = grid[i][j]
        attacked = False
        for di, dj in dirs:
            if attacked:
                break
            x, y = i + di, j + dj
            gap = []
            while 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0:
                gap.append((x, y))
                x += di
                y += dj
            if not (0 <= x < rows and 0 <= y < cols):
                continue
            hx, hy = x, y
            if grid[hx][hy] == 0 or grid[hx][hy] == T:
                continue
            hit_pos = (hx, hy)
            hit_cid = pos_to_comp.get(hit_pos, -1)
            if hit_cid == -1 or len(components[hit_cid]) < 2:
                continue
            large_comp = components[hit_cid]
            C = grid[hx][hy]
            gap_size = len(gap)
            if gap_size == 0:
                continue
            # fill gap with C
            for px, py in gap:
                new_grid[px][py] = C
            # conquer
            comp_set = set(large_comp)
            dist = {}
            q = deque([hit_pos])
            dist[hit_pos] = 0
            while q:
                cx, cy = q.popleft()
                for dx, dy in dirs:
                    nx, ny = cx + dx, cy + dy
                    npos = (nx, ny)
                    if 0 <= nx < rows and 0 <= ny < cols and npos in comp_set and npos not in dist:
                        dist[npos] = dist[(cx, cy)] + 1
                        q.append(npos)
            other_pos = [p for p in large_comp if p != hit_pos]
            if not other_pos:
                continue
            dist_list = [(dist.get(p, 0), p) for p in other_pos]
            dist_list.sort(key=lambda z: -z[0])
            num_to_change = min(gap_size, len(dist_list))
            for k in range(num_to_change):
                _, p = dist_list[k]
                new_grid[p[0]][p[1]] = T
            attacked = True
    return new_grid