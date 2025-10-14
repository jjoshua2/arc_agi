import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    output = grid.copy()
    
    # Directions for diagonal propagation: up-left, up-right, down-left, down-right
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Find all source cells (non-zero and not purple)
    sources = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and grid[r, c] != 8:
                sources.append((r, c, grid[r, c]))
    
    # Propagate from each source
    for r, c, color in sources:
        for dr, dc in directions:
            # Propagate in this direction until we hit boundary or non-empty cell
            step = 1
            while True:
                nr, nc = r + dr * step, c + dc * step
                
                # Check if out of bounds
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    # Mark the boundary cell with red (2)
                    # Find the last valid cell in this direction
                    last_r, last_c = r + dr * (step - 1), c + dc * (step - 1)
                    if 0 <= last_r < h and 0 <= last_c < w:
                        if output[last_r, last_c] == 0:
                            output[last_r, last_c] = 2
                    break
                
                # Check if cell is already occupied (non-zero)
                if grid[nr, nc] != 0:
                    break
                
                # Fill the cell with the source color
                output[nr, nc] = color
                step += 1
    
    return output.tolist()