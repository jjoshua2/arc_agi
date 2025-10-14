import numpy as np
from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
        
    grid_np = np.array(grid)
    rows, cols = grid_np.shape
    output = grid_np.copy()
    
    # Directions for 4-connected flood fill
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Find all source cells (non-zero, non-8)
    sources = []
    for r in range(rows):
        for c in range(cols):
            if grid_np[r, c] != 0 and grid_np[r, c] != 8:
                sources.append((r, c, grid_np[r, c]))
    
    # For each source, perform flood fill in all four directions
    for sr, sc, color in sources:
        # Skip if this is already the target color or purple
        if output[sr, sc] == color or output[sr, sc] == 8:
            continue
            
        # Perform BFS flood fill from this source
        visited = set()
        queue = deque([(sr, sc)])
        visited.add((sr, sc))
        
        while queue:
            r, c = queue.popleft()
            
            # Set the color
            output[r, c] = color
            
            # Check all four directions
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    (nr, nc) not in visited and 
                    output[nr, nc] != 8 and  # Don't fill over purple
                    output[nr, nc] != color):  # Don't revisit same color
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    return output.tolist()