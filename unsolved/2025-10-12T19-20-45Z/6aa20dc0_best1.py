import numpy as np
from typing import List, Tuple, Set
from collections import Counter, deque

def find_components(grid: List[List[int]], color: int, h: int, w: int) -> List[List[Tuple[int, int]]]:
    visited = [[False] * w for _ in range(h)]
    components = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == color and not visited[i][j]:
                comp = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == color and not visited[nx][ny]:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                components.append(comp)
    return components

def get_min_max(comp: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    if not comp:
        return 0, 0, 0, 0
    minr = min(r for r, c in comp)
    maxr = max(r for r, c in comp)
    minc = min(c for r, c in comp)
    maxc = max(c for r, c in comp)
    return minr, maxr, minc, maxc

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    h = len(grid_lst)
    w = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    all_cells = [cell for row in grid_lst for cell in row]
    bg_count = Counter(all_cells)
    if not bg_count:
        return output
    bg = bg_count.most_common(1)[0][0]
    # find base 3x3
    base_r, base_c = -1, -1
    max_distinct = -1
    best_score = -1
    for i in range(h - 2):
        for j in range(w - 2):
            colors = set()
            freq = Counter()
            count_non_bg = 0
            for di in range(3):
                for dj in range(3):
                    cell = grid_lst[i + di][j + dj]
                    if cell != bg:
                        colors.add(cell)
                        freq[cell] += 1
                        count_non_bg += 1
            num_dist = len(colors)
            score = num_dist * 10 + count_non_bg
            if num_dist > max_distinct or (num_dist == max_distinct and score > best_score):
                max_distinct = num_dist
                best_score = score
                base_r, base_c = i, j
    if base_r == -1 or max_distinct < 3:
        return output
    # extract 3x3
    three = [[grid_lst[base_r + dr][base_c + dc] for dc in range(3)] for dr in range(3)]
    # F
    freq = Counter(cell for row in three for cell in row if cell != bg)
    if not freq:
        return output
    F = max(freq, key=freq.get)
    # marker colors
    marker_set = {col for col in freq if col != F and freq[col] > 0}
    if len(marker_set) != 2:
        return output
    marker_colors = list(marker_set)
    # small_top_color
    pos = {}
    for col in marker_colors:
        pos[col] = [(dr, dc) for dr in range(3) for dc in range(3) if three[dr][dc] == col]
    avg_r = {}
    for col in marker_colors:
        if pos[col]:
            avg_r[col] = sum(dr for dr, _ in pos[col]) / len(pos[col])
        else:
            avg_r[col] = float('inf')
    small_top_color = min(marker_colors, key=lambda col: avg_r[col])
    small_bottom_color = [c for c in marker_colors if c != small_top_color][0]
    small_top_pos = pos[small_top_color]
    small_top_avg_col = sum(dc for _, dc in small_top_pos) / len(small_top_pos) if small_top_pos else 0
    small_top_is_left = small_top_avg_col < 1.5
    # fill_mask
    fill_mask: Set[Tuple[int, int]] = {(dr, dc) for dr in range(3) for dc in range(3) if three[dr][dc] == F}
    # components
    comps = {}
    for col in marker_colors:
        comps[col] = find_components(grid_lst, col, h, w)
    # pairs
    for idx1 in range(len(marker_colors)):
        col1 = marker_colors[idx1]
        col2 = marker_colors[1 - idx1]
        for comp1 in comps[col1]:
            for comp2 in comps[col2]:
                minr1, maxr1, _, _ = get_min_max(comp1)
                minr2, maxr2, _, _ = get_min_max(comp2)
                if maxr1 >= minr2:
                    continue
                min_r = minr1
                max_r = maxr2
                hh = max_r - min_r + 1
                _, _, minc1, maxc1 = get_min_max(comp1)
                _, _, minc2, maxc2 = get_min_max(comp2)
                min_cc = min(minc1, minc2)
                max_cc = max(maxc1, maxc2)
                ww = max_cc - min_cc + 1
                if hh == ww and hh % 3 == 0 and hh >= 3:
                    s = hh // 3
                    if hh == 3 and abs(min_r - base_r) < 3 and abs(min_cc - base_c) < 3:
                        continue  # skip base area
                    top_color = col1
                    v_flip = (top_color != small_top_color)
                    # large top avg rel col
                    num_top = len(comp1)
                    sum_rel_c = sum((c - min_cc) for r, c in comp1)
                    large_top_avg_rel = sum_rel_c / num_top
                    center = (3 * s - 1) / 2.0
                    large_top_is_left = large_top_avg_rel < center
                    h_flip = (large_top_is_left != small_top_is_left)
                    # transform mask
                    current_mask = fill_mask.copy()
                    if h_flip:
                        current_mask = {(dr, 2 - dc) for dr, dc in current_mask}
                    if v_flip:
                        current_mask = {(2 - dr, dc) for dr, dc in current_mask}
                    # scale and fill
                    for dr, dc in current_mask:
                        r_start = min_r + s * dr
                        c_start = min_cc + s * dc
                        for ii in range(s):
                            for jj in range(s):
                                rr = r_start + ii
                                cc = c_start + jj
                                if 0 <= rr < h and 0 <= cc < w and output[rr][cc] == bg:
                                    output[rr][cc] = F
    return output