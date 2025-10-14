import math
from collections import deque, Counter

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[False] * cols for _ in range(rows)]
    comp_id = [[-1] * cols for _ in range(rows)]
    component_lists = []
    comp_color = []
    comp_size = []
    current_id = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] > 0 and not visited[i][j]:
                color = grid[i][j]
                q = deque([(i, j)])
                visited[i][j] = True
                comp_list = [(i, j)]
                comp_id[i][j] = current_id
                while q:
                    x, y = q.popleft()
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                            comp_list.append((nx, ny))
                            comp_id[nx][ny] = current_id
                component_lists.append(comp_list)
                comp_color.append(color)
                comp_size.append(len(comp_list))
                current_id += 1
    output = [row[:] for row in grid]
    sqrt2 = math.sqrt(2)
    for cid in range(current_id):
        if comp_size[cid] > 2:
            continue
        positions = component_lists[cid]
        if comp_size[cid] == 1:
            r, c = positions[0]
            adj_counts = Counter()
            for dx, dy in dirs:
                nr = r + dx
                nc = c + dy
                if 0 <= nr < rows and 0 <= nc < cols:
                    aid = comp_id[nr][nc]
                    if aid != -1:
                        adj_counts[aid] += 1
            max_cnt = max(adj_counts.values() or [0])
            if max_cnt < 2:
                output[r][c] = 0
                continue
            candidates = [aid for aid, acnt in adj_counts.items() if acnt == max_cnt]
            best_cid = max(candidates, key=lambda x: (comp_size[x], comp_color[x]))
            C = comp_color[best_cid]
            pos_list = component_lists[best_cid]
            n = len(pos_list)
            sumr = sum(rr for rr, _ in pos_list)
            sumc = sum(cc for _, cc in pos_list)
            cent_r = sumr / n
            cent_c = sumc / n
            dist = math.sqrt((r - cent_r)**2 + (c - cent_c)**2)
            if dist > sqrt2:
                output[r][c] = C
            else:
                output[r][c] = 0
        else:  # size == 2
            adj_counts = Counter()
            for pr, pc in positions:
                for dx, dy in dirs:
                    nr = pr + dx
                    nc = pc + dy
                    if 0 <= nr < rows and 0 <= nc < cols:
                        aid = comp_id[nr][nc]
                        if aid != -1 and aid != cid:
                            adj_counts[aid] += 1
            max_cnt = max(adj_counts.values() or [0])
            if max_cnt < 3:
                for pr, pc in positions:
                    output[pr][pc] = 0
            else:
                candidates = [aid for aid, acnt in adj_counts.items() if acnt == max_cnt]
                best_cid = max(candidates, key=lambda x: (comp_size[x], comp_color[x]))
                C = comp_color[best_cid]
                for pr, pc in positions:
                    output[pr][pc] = C
    return output