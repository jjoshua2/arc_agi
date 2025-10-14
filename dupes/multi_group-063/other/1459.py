import sys

def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    if rows == 0:
        return []
    cols = len(grid[0])
    best_area = sys.maxsize
    best_p = sys.maxsize
    best_q = sys.maxsize
    best_tile = None
    MAX_PERIOD = 15
    for P in range(1, MAX_PERIOD + 1):
        for Q in range(1, MAX_PERIOD + 1):
            tile = [[-1] * Q for _ in range(P)]
            valid = True
            for r in range(rows):
                if not valid:
                    break
                for c in range(cols):
                    if grid[r][c] != 0:
                        pr = r % P
                        pc = c % Q
                        if tile[pr][pc] == -1:
                            tile[pr][pc] = grid[r][c]
                        elif tile[pr][pc] != grid[r][c]:
                            valid = False
                            break
            if valid:
                filled = sum(1 for i in range(P) for j in range(Q) if tile[i][j] != -1)
                area = P * Q
                if filled == area:
                    update = False
                    if area < best_area:
                        update = True
                    elif area == best_area:
                        if P < best_p or (P == best_p and Q < best_q):
                            update = True
                    if update:
                        best_area = area
                        best_p = P
                        best_q = Q
                        best_tile = [row[:] for row in tile]
    if best_tile is None:
        return [row[:] for row in grid]
    output = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                pr = r % best_p
                pc = c % best_q
                output[r][c] = best_tile[pr][pc]
    return output