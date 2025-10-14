from typing import List, Tuple

def get_zero_components(grid: List[List[int]]) -> List[Tuple[List[Tuple[int, int]], int]]:
    H = len(grid)
    if H == 0:
        return []
    W = len(grid[0])
    visited = [[False] * W for _ in range(H)]
    components = []
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 0 and not visited[i][j]:
                comp = []
                stack: List[Tuple[int, int]] = [(i, j)]
                visited[i][j] = True
                color = grid[i][j]  # 0
                
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                
                components.append((comp, color))
    
    return components

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    H, W = len(grid_lst), len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    zero_components = get_zero_components(grid_lst)
    
    for comp, color in zero_components:
        touches_border = any(
            r == 0 or r == H - 1 or c == 0 or c == W - 1
            for r, c in comp
        )
        if not touches_border:
            for r, c in comp:
                output[r][c] = 4
    
    return output