from typing import List, Tuple
from collections import Counter

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    
    # Find full rows
    full_rows: List[Tuple[int, int]] = []
    for r in range(rows):
        counts = Counter(grid[r])
        max_count = max(counts.get(c, 0) for c in range(1, 10))
        if max_count >= cols - 2:
            for c in range(1, 10):
                if counts.get(c, 0) == max_count:
                    full_rows.append((r, c))
                    break
    if len(full_rows) != 2:
        raise ValueError("Expected exactly two full rows")
    full_list = sorted(full_rows)
    T, C_top = full_list[0]
    B, C_bottom = full_list[1]
    
    H = B - T + 1
    ignored = {T, B}
    num_non_ignored = rows - 2
    
    # Find uniform columns ignoring full rows
    uniform_cols: List[Tuple[int, int]] = []
    for c in range(cols):
        vals = [grid[r][c] for r in range(rows) if r not in ignored]
        if len(vals) == num_non_ignored and len(set(vals)) == 1 and vals[0] > 0:
            uniform_cols.append((c, vals[0]))
    if len(uniform_cols) != 2:
        raise ValueError("Expected exactly two uniform columns")
    sorted_uniform = sorted(uniform_cols)
    CL, L = sorted_uniform[0]
    CR, R = sorted_uniform[-1]
    
    W_mid = CR - CL - 1
    out: List[List[int]] = [[0] * (W_mid + 2) for _ in range(H)]
    
    # Set rails
    for rr in range(H):
        out[rr][0] = L
        out[rr][W_mid + 1] = R
    
    # Set top and bottom full
    for cc in range(1, W_mid + 1):
        out[0][cc] = C_top
        out[H - 1][cc] = C_bottom
    
    # Find S
    possible_s = set()
    for i in range(T + 1, B):
        for c in range(CL + 1, CR):
            v = grid[i][c]
            if v > 0:
                possible_s.add(v)
    if len(possible_s) != 1:
        raise ValueError("Expected exactly one scattered color")
    S = next(iter(possible_s))
    
    # Original positions for each between kk=1 to H-2
    between_h = H - 2
    if between_h < 0:
        return out
    orig: List[set] = [set() for _ in range(between_h + 1)]  # index 0 unused
    for kk in range(1, between_h + 1):
        i = T + kk
        for rel in range(W_mid):
            cc = CL + 1 + rel
            if grid[i][cc] == S:
                orig[kk].add(rel)
    
    # Horizontal fills
    for kk in range(1, H - 1):
        if orig[kk]:
            rels = list(orig[kk])
            minr = min(rels)
            maxr = max(rels)
            extend_l = (L == S)
            extend_r = (R == S)
            fill_min = 0 if extend_l else minr
            fill_max = W_mid - 1 if extend_r else maxr
            if extend_l or extend_r:
                # Full span fill
                for rel in range(fill_min, fill_max + 1):
                    out[kk][1 + rel] = S
            else:
                # Just originals
                for rel in rels:
                    out[kk][1 + rel] = S
    
    # Prepare for vertical
    min_k_per_rel = [H] * W_mid
    max_k_per_rel = [0] * W_mid
    for kk in range(1, H - 1):
        for rel in orig[kk]:
            min_k_per_rel[rel] = min(min_k_per_rel[rel], kk)
            max_k_per_rel[rel] = max(max_k_per_rel[rel], kk)
    
    # Upward vertical if top == S
    if C_top == S:
        for rel in range(W_mid):
            if max_k_per_rel[rel] > 0:
                for kkk in range(1, max_k_per_rel[rel] + 1):
                    out[kkk][1 + rel] = S
    
    # Downward vertical if bottom == S
    if C_bottom == S:
        for rel in range(W_mid):
            if min_k_per_rel[rel] < H:
                for kkk in range(min_k_per_rel[rel], H - 1):
                    out[kkk][1 + rel] = S
    
    return out