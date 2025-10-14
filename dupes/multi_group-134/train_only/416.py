from collections import deque
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    out = copy.deepcopy(grid)
    
    def is_connected(cells):
        if not cells:
            return True
        cell_set = set(cells)
        start = cells[0]
        visited = set()
        q = deque([start])
        visited.add(start)
        while q:
            cr, cc = q.popleft()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    nt = (nr, nc)
                    if nt in cell_set and nt not in visited:
                        visited.add(nt)
                        q.append(nt)
        return len(visited) == len(cells)
    
    for sr in range(h - 2):
        for sc in range(w - 2):
            non_zero_cells = []
            for dr in range(3):
                r = sr + dr
                for dc in range(3):
                    c = sc + dc
                    if grid[r][c] != 0:
                        non_zero_cells.append((r, c))
            if not non_zero_cells:
                continue
            minr = min(rr for rr, _ in non_zero_cells)
            maxr = max(rr for rr, _ in non_zero_cells)
            minc = min(cc for _, cc in non_zero_cells)
            maxc = max(cc for _, cc in non_zero_cells)
            if minr != sr or maxr != sr + 2 or minc != sc or maxc != sc + 2:
                continue
            if not is_connected(non_zero_cells):
                continue
            # Draw frame
            top_r = sr - 1
            bottom_r = sr + 3
            left_c = sc - 1
            right_c = sc + 3
            # Top horizontal
            if top_r >= 0:
                for cc in range(max(0, left_c), min(w, right_c + 1)):
                    out[top_r][cc] = 5
            # Bottom horizontal
            if bottom_r < h:
                for cc in range(max(0, left_c), min(w, right_c + 1)):
                    out[bottom_r][cc] = 5
            # Left vertical
            if left_c >= 0:
                for rr in range(max(0, top_r), min(h, bottom_r + 1)):
                    out[rr][left_c] = 5
            # Right vertical
            if right_c < w:
                for rr in range(max(0, top_r), min(h, bottom_r + 1)):
                    out[rr][right_c] = 5
    
    return out