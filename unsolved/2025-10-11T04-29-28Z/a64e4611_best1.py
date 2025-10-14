from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = grid_lst  # read-only reference to input values

    visited = [[False]*cols for _ in range(rows)]
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    largest_comp = []
    # Find all connected components of zeros (4-connected)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                q = deque()
                q.append((r,c))
                visited[r][c] = True
                comp = [(r,c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in directions:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                            visited[nr][nc] = True
                            q.append((nr,nc))
                            comp.append((nr,nc))
                if len(comp) > len(largest_comp):
                    largest_comp = comp

    # If no zero component found, return copy of original
    output = [row[:] for row in grid_lst]
    if not largest_comp:
        return output

    # Fill largest zero-component with green (color 3)
    for (r,c) in largest_comp:
        output[r][c] = 3

    return output