from collections import deque, Counter
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    flat = [grid[i][j] for i in range(rows) for j in range(cols)]
    count = Counter(flat)
    if 0 in count:
        del count[0]
    if not count:
        return [row[:] for row in grid]
    C = max(count, key=count.get)
    
    # Original C positions
    original_C_pos = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == C]
    
    # Find C components bbs
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()
    components = []
    for si in range(rows):
        for sj in range(cols):
            if grid[si][sj] == C and (si, sj) not in visited:
                q = deque([(si, sj)])
                visited.add((si, sj))
                comp_pos = [(si, sj)]
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == C and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            q.append((nx, ny))
                            comp_pos.append((nx, ny))
                minr = min(p[0] for p in comp_pos)
                maxr = max(p[0] for p in comp_pos)
                minc_ = min(p[1] for p in comp_pos)
                maxc_ = max(p[1] for p in comp_pos)
                components.append((minr, maxr, minc_, maxc_))
    
    # Candidates
    candidates = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and grid[i][j] != C:
                adj_c = any(0 <= i + dx < rows and 0 <= j + dy < cols and grid[i + dx][j + dy] == C
                            for dx, dy in directions)
                if not adj_c:
                    candidates.append((i, j, grid[i][j]))
    
    # Kept
    kept = []
    for pos in candidates:
        r, c, k = pos
        discard = any(minc_ <= c <= maxc_ and r <= maxr for _, maxr, minc_, maxc_ in components)
        if not discard:
            kept.append((r, c, k))
    
    # Output init
    output = [row[:] for row in grid]
    for r, c in original_C_pos:
        output[r][c] = 0
    for pos in candidates:
        if pos not in kept:
            r, c, _ = pos
            output[r][c] = 0
    
    if not kept:
        return output
    
    # Build initial graph with union find
    num_t = len(kept)
    parent = list(range(num_t))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px = find(x)
        py = find(y)
        if px != py:
            parent[px] = py
    for i in range(num_t):
        for j in range(i + 1, num_t):
            r1, c1 = kept[i][0], kept[i][1]
            r2, c2 = kept[j][0], kept[j][1]
            if r1 == r2 and abs(c1 - c2) <= 5:
                union(i, j)
            elif c1 == c2 and 1 <= abs(r1 - r2) <= 2:
                union(i, j)
    
    # Initial comps
    initial_comps = {}
    for i in range(num_t):
        p = find(i)
        initial_comps.setdefault(p, []).append(i)
    initial_comp_lists = list(initial_comps.values())
    
    # Full clusters
    clusters = []
    for init_list in initial_comp_lists:
        current_indices = set(init_list)
        targets_in = [(kept[idx][0], kept[idx][1]) for idx in current_indices]
        if not targets_in:
            continue
        base_r = max(tr for tr, _ in targets_in)
        min_c = min(tc for _, tc in targets_in)
        max_c = max(tc for _, tc in targets_in)
        added = True
        while added:
            added = False
            for ti in range(num_t):
                if ti in current_indices:
                    continue
                tr, tc = kept[ti][0], kept[ti][1]
                if tr >= base_r or base_r - tr > 2:
                    continue
                if min_c - 1 <= tc <= max_c + 1:
                    current_indices.add(ti)
                    added = True
                    min_c = min(min_c, tc)
                    max_c = max(max_c, tc)
        cluster = [kept[ti] for ti in current_indices]
        if cluster:
            clusters.append(cluster)
    
    # Order clusters
    def cl_key(clu):
        min_r_cl = min(t[0] for t in clu)
        min_c_cl = min(t[1] for t in clu)
        return (min_r_cl, min_c_cl)
    clusters.sort(key=cl_key)
    num_clusters = len(clusters)
    
    # Assign n
    n_list = [0] * num_clusters
    for or_, oc in original_C_pos:
        dists = []
        for i_cl in range(num_clusters):
            d = min(abs(or_ - tr) + abs(oc - tc) for tr, tc, _ in clusters[i_cl])
            dists.append(d)
        if not dists:
            continue
        min_d = min(dists)
        candidates_i = [i for i in range(num_clusters) if dists[i] == min_d]
        best_i = min(candidates_i)
        n_list[best_i] += 1
    
    # Now process each cluster
    for i_cl in range(num_clusters):
        cluster = clusters[i_cl]
        n = n_list[i_cl]
        if n == 0:
            continue
        m = len(cluster)
        if m == 0:
            continue
        
        # target_rows_dict row -> list c
        target_rows_dict = {}
        for tr, tc, _ in cluster:
            target_rows_dict.setdefault(tr, []).append(tc)
        
        base_r = max(target_rows_dict.keys())
        
        # h
        base_cols_set = set(target_rows_dict.get(base_r, []))
        base_cols_list = sorted(base_cols_set)
        h = 0
        if len(base_cols_list) >= 2:
            for jj in range(len(base_cols_list) - 1):
                h += base_cols_list[jj + 1] - base_cols_list[jj] - 1
        # set h fills
        for jj in range(len(base_cols_list) - 1):
            for cc in range(base_cols_list[jj] + 1, base_cols_list[jj + 1]):
                output[base_r][cc] = C
        
        # v
        v = 0
        for tr, tc, _ in cluster:
            if tr >= base_r:
                continue
            for rr in range(tr + 1, base_r):
                if output[rr][tc] == 0:
                    output[rr][tc] = C
                    v += 1
        
        min_used = h + v
        remaining = n - min_used
        if remaining < 0:
            remaining = 0  # shouldn't happen
        
        has_off_row = any(tr < base_r for tr, _, _ in cluster)
        v_total = v
        
        if v_total > 0:
            # horizontal extensions
            target_rows = list(target_rows_dict.keys())
            such_rows = []
            for rr in target_rows:
                cols_r_set = set(target_rows_dict[rr])
                cols_r_list = sorted(cols_r_set)
                h_r = 0
                if len(cols_r_list) >= 2:
                    for jj in range(len(cols_r_list) - 1):
                        h_r += cols_r_list[jj + 1] - cols_r_list[jj] - 1
                if h_r == 0:
                    such_rows.append(rr)
            such_rows.sort()
            for ii, rr in enumerate(such_rows):
                cols_r_set = set(target_rows_dict[rr])
                # assume single
                tc = min(cols_r_set)  # or average if multiple
                primary_right = (ii % 2 == 0)
                deltas = [1 if primary_right else -1]
                if not primary_right:
                    deltas = [-1, 1]
                added_this = False
                for d_idx in range(2):
                    if remaining <= 0:
                        break
                    delta = 1 if (primary_right and d_idx == 0) or (not primary_right and d_idx == 1) else -1
                    pos_c = tc + delta
                    set_pos = None
                    for step in range(1, cols + 1):
                        p_c = tc + step * delta
                        if 0 <= p_c < cols and output[rr][p_c] == 0:
                            set_pos = p_c
                            break
                        if step > cols:
                            break
                    if set_pos is not None:
                        output[rr][set_pos] = C
                        remaining -= 1
                        added_this = True
                    if d_idx == 0 and added_this:
                        break  # if primary added, skip other
        else:
            # v_total == 0
            if has_off_row:
                # vertical 1 down 1 up per tc l to r
                target_cols_set = set(tc for _, tc, _ in cluster)
                target_cols = sorted(target_cols_set)
                for tc in target_cols:
                    # down
                    if remaining > 0 and base_r + 1 < rows and output[base_r + 1][tc] == 0:
                        output[base_r + 1][tc] = C
                        remaining -= 1
                    # up
                    if remaining > 0 and base_r - 1 >= 0 and output[base_r - 1][tc] == 0:
                        output[base_r - 1][tc] = C
                        remaining -= 1
            else:
                # no off_row v=0
                min_tc = min(tc for _, tc, _ in cluster)
                max_tc = max(tc for _, tc, _ in cluster)
                # down min
                if remaining > 0 and base_r + 1 < rows and output[base_r + 1][min_tc] == 0:
                    output[base_r + 1][min_tc] = C
                    remaining -= 1
                # up max
                if remaining > 0 and base_r - 1 >= 0 and output[base_r - 1][max_tc] == 0:
                    output[base_r - 1][max_tc] = C
                    remaining -= 1
                # right horizontal
                if remaining > 0 and max_tc + 1 < cols and output[base_r][max_tc + 1] == 0:
                    output[base_r][max_tc + 1] = C
                    remaining -= 1
                # further if needed, e.g.
                if remaining > 0 and base_r + 1 < rows and output[base_r + 1][max_tc] == 0:
                    output[base_r + 1][max_tc] = C
                    remaining -= 1
                if remaining > 0 and base_r - 1 >= 0 and output[base_r - 1][min_tc] == 0:
                    output[base_r - 1][min_tc] = C
                    remaining -= 1
                # etc, but assume sufficient for examples
    
    return output