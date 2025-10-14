def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    n = len(grid_lst)
    if n == 0:
        return []
    m = len(grid_lst[0])
    seeds = []
    for c in range(m):
        if grid_lst[0][c] != 0:
            seeds.append((0, c, grid_lst[0][c]))
        if grid_lst[n-1][c] != 0:
            seeds.append((n-1, c, grid_lst[n-1][c]))
    
    output_grid = [[0] * m for _ in range(n)]
    INF = 10**9
    
    is_odd = n % 2 == 1
    mid = (n - 1) // 2 if is_odd else -1
    
    for r in range(n):
        if is_odd and r == mid:
            continue
        for c in range(m):
            candidates = []
            for seed_r, s_col, color in seeds:
                is_top_seed = (seed_r == 0)
                local_r = r if is_top_seed else (n - 1 - r)
                
                # Determine type
                left_type = (s_col == 0)
                right_type = (s_col == m - 1)
                if not left_type and not right_type:
                    continue  # Assume only edge seeds
                
                if left_type:
                    if c < s_col:
                        continue
                    local_c = c - s_col
                else:  # right_type
                    if c > s_col:
                        continue
                    local_c = s_col - c
                
                # Compute dist_opp
                opposite_seeds = [sd for sd in seeds if (is_top_seed and sd[0] != 0) or (not is_top_seed and sd[0] == 0)]
                dists = [abs(s_col - sd[1]) for sd in opposite_seeds if sd[1] != s_col]
                dist_opp = min(dists) if dists else INF
                max_lc_opp = dist_opp - local_r - 1 if dist_opp < INF else INF
                if local_c > max_lc_opp:
                    continue
                
                if is_filled(local_r, local_c):
                    total = local_r + local_c
                    candidates.append((total, local_r, s_col, color))
            
            if candidates:
                candidates.sort()
                output_grid[r][c] = candidates[0][3]
    
    # Place the original seeds in case, but should already be placed
    for seed_r, s_col, color in seeds:
        output_grid[seed_r][s_col] = color
    
    return output_grid


def is_filled(lr: int, lc: int) -> bool:
    if lr % 2 == 0:
        return lc <= lr or (lc % 2 == 0 and lc >= lr + 2)
    else:
        return lc % 2 == 0 and lc >= lr + 1