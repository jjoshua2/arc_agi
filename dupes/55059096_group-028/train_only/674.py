def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    h = len(grid_lst)
    if h == 0:
        return []
    w = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    # Find centers of pluses
    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (grid_lst[r][c] == 3 and
                grid_lst[r - 1][c] == 3 and
                grid_lst[r + 1][c] == 3 and
                grid_lst[r][c - 1] == 3 and
                grid_lst[r][c + 1] == 3):
                centers.append((r, c))
    
    # Connect pairs where |dr| == |dc| > 0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            r1, c1 = centers[i]
            r2, c2 = centers[j]
            if r1 > r2:
                r1, r2 = r2, r1
                c1, c2 = c2, c1
            dr = r2 - r1
            dc = c2 - c1
            if dr > 0 and abs(dc) == dr:
                s = 1 if dc > 0 else -1
                for k in range(1, dr):
                    tr = r1 + k
                    tc = c1 + k * s
                    if 0 <= tc < w:
                        output[tr][tc] = 2
    
    return output