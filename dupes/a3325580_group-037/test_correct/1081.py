import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i,j] != 0 and not visited[i,j]:
                color = grid[i,j]
                stack = [(i,j)]
                visited[i,j] = True
                size = 1
                min_col = j
                cells = [(i,j)]  # to update min_col properly
                while stack:
                    x,y = stack.pop()
                    min_col = min(min_col, y)
                    for dx,dy in dirs:
                        nx,ny = x+dx, y+dy
                        if 0<=nx<rows and 0<=ny<cols and not visited[nx,ny] and grid[nx,ny] == color:
                            visited[nx,ny] = True
                            stack.append((nx,ny))
                            size +=1
                            min_col = min(min_col, ny)
                components.append( (min_col, color, size) )
    if not components:
        return []
    max_s = max(s for _,_,s in components)
    max_comps = [ (mc, c) for mc,c,s in components if s == max_s ]
    max_comps.sort(key=lambda x: x[0])  # sort by min_col
    K = len(max_comps)
    h = max_s
    out = [[max_comps[k][1] for k in range(K)] for _ in range(h)]
    return out