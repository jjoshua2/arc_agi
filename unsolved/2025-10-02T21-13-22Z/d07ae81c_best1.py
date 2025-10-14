import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Create a copy to modify
    result = grid.copy()
    
    # Find special colored cells (non-zero values that are not part of the main background)
    # We'll consider cells that are different from their neighbors as special
    special_cells = []
    
    # For simplicity, let's find cells that are not part of large uniform regions
    # We'll use a simple approach: cells that have a value different from most of their neighbors
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0:
                # Check if this cell is different from its neighbors
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            neighbors.append(grid[ni, nj])
                
                if neighbors and grid[i, j] not in neighbors:
                    special_cells.append((i, j, grid[i, j]))
    
    # For each special cell, propagate its color along the diagonal
    for i, j, color in special_cells:
        # Propagate in all four diagonal directions
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for di, dj in directions:
            ci, cj = i, j
            while True:
                ci += di
                cj += dj
                if not (0 <= ci < h and 0 <= cj < w):
                    break
                if grid[ci, cj] != 0:  # Only change non-zero cells
                    result[ci, cj] = color
    
    return result.tolist()