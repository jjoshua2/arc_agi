from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    orig = [row[:] for row in grid]
    out = [row[:] for row in grid]

    Y = 4  # yellow
    BLUE = 1
    GREEN = 3

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def count_adjacent_yellow(r: int, c: int) -> int:
        cnt = 0
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if inb(nr, nc) and orig[nr][nc] == Y:
                cnt += 1
        return cnt

    # Collect horizontal runs of 4's: (row, c0, c1)
    horiz_runs: List[Tuple[int,int,int]] = []
    for r in range(h):
        c = 0
        while c < w:
            if orig[r][c] == Y:
                c0 = c
                while c < w and orig[r][c] == Y:
                    c += 1
                c1 = c - 1
                horiz_runs.append((r, c0, c1))
            else:
                c += 1

    # Collect vertical runs of 4's: (col, r0, r1)
    vert_runs: List[Tuple[int,int,int]] = []
    for c in range(w):
        r = 0
        while r < h:
            if orig[r][c] == Y:
                r0 = r
                while r < h and orig[r][c] == Y:
                    r += 1
                r1 = r - 1
                vert_runs.append((c, r0, r1))
            else:
                r += 1

    # Helper: infer phase for a horizontal run
    # Returns: (top_color, top_parity, bottom_color, bottom_parity)
    def infer_phase_horizontal(r: int, c0: int, c1: int):
        L = c1 - c0 + 1
        top_hint = None  # (color, parity)
        bot_hint = None
        # scan for hints
        for i in range(L):
            c = c0 + i
            if r-1 >= 0:
                v = orig[r-1][c]
                if v in (BLUE, GREEN):
                    top_hint = (v, i % 2)
                    break
        for i in range(L):
            c = c0 + i
            if r+1 < h:
                v = orig[r+1][c]
                if v in (BLUE, GREEN):
                    bot_hint = (v, i % 2)
                    break
        # derive phase
        if top_hint:
            top_color, top_parity = top_hint
            bottom_color = GREEN if top_color == BLUE else BLUE
            bottom_parity = 1 - top_parity
        elif bot_hint:
            bottom_color, bottom_parity = bot_hint
            top_color = GREEN if bottom_color == BLUE else BLUE
            top_parity = 1 - bottom_parity
        else:
            # canonical default: top GREEN on first (even index), bottom BLUE on odd
            top_color, top_parity = GREEN, 0
            bottom_color, bottom_parity = BLUE, 1
        return top_color, top_parity, bottom_color, bottom_parity

    # Helper: infer phase for a vertical run
    # Returns: (left_color, left_parity, right_color, right_parity)
    def infer_phase_vertical(c: int, r0: int, r1: int):
        L = r1 - r0 + 1
        left_hint = None
        right_hint = None
        for j in range(L):
            r = r0 + j
            if c-1 >= 0:
                v = orig[r][c-1]
                if v in (BLUE, GREEN):
                    left_hint = (v, j % 2)
                    break
        for j in range(L):
            r = r0 + j
            if c+1 < w:
                v = orig[r][c+1]
                if v in (BLUE, GREEN):
                    right_hint = (v, j % 2)
                    break
        if left_hint:
            left_color, left_parity = left_hint
            right_color = GREEN if left_color == BLUE else BLUE
            right_parity = 1 - left_parity
        elif right_hint:
            right_color, right_parity = right_hint
            left_color = GREEN if right_color == BLUE else BLUE
            left_parity = 1 - right_parity
        else:
            # canonical default: left GREEN on first (even index), right BLUE on odd
            left_color, left_parity = GREEN, 0
            right_color, right_parity = BLUE, 1
        return left_color, left_parity, right_color, right_parity

    # Decorate horizontal runs
    for r, c0, c1 in horiz_runs:
        top_color, top_parity, bottom_color, bottom_parity = infer_phase_horizontal(r, c0, c1)
        L = c1 - c0 + 1
        for i in range(L):
            c = c0 + i
            # top
            tr, tc = r-1, c
            if tr >= 0 and i % 2 == top_parity:
                if out[tr][tc] == 0 and count_adjacent_yellow(tr, tc) == 1:
                    out[tr][tc] = top_color
            # bottom
            br, bc = r+1, c
            if br < h and i % 2 == bottom_parity:
                if out[br][bc] == 0 and count_adjacent_yellow(br, bc) == 1:
                    out[br][bc] = bottom_color

    # Decorate vertical runs
    for c, r0, r1 in vert_runs:
        left_color, left_parity, right_color, right_parity = infer_phase_vertical(c, r0, r1)
        L = r1 - r0 + 1
        for j in range(L):
            r = r0 + j
            # left
            lr, lc = r, c-1
            if lc >= 0 and j % 2 == left_parity:
                if out[lr][lc] == 0 and count_adjacent_yellow(lr, lc) == 1:
                    out[lr][lc] = left_color
            # right
            rr, rc = r, c+1
            if rc < w and j % 2 == right_parity:
                if out[rr][rc] == 0 and count_adjacent_yellow(rr, rc) == 1:
                    out[rr][rc] = right_color

    return out