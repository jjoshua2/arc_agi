def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    h = len(grid_lst)
    w = len(grid_lst[0])
    output = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid_lst[i][j] != 0:
                # Check if (i,j) is on the boundary: has a neighbor that is out of bounds or has value 0
                on_boundary = False
                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                    ni, nj = i+di, j+dj
                    if ni < 0 or ni >= h or nj < 0 or nj >= w:
                        on_boundary = True
                    elif grid_lst[ni][nj] == 0:
                        on_boundang = True
                if on_boundary:
                    output[i][j] = 8
    return output