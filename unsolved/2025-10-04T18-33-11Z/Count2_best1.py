import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    
    # Find all cells with value 1
    ones_positions = np.argwhere(grid == 1)
    
    # If no ones found, return empty list
    if len(ones_positions) == 0:
        return []
    
    # Count corners
    corner_count = 0
    rows, cols = grid.shape
    
    # Directions for 4-connected neighbors
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r, c in ones_positions:
        # Get 4-connected neighbors
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                neighbors.append((nr, nc))
        
        # Check if this is a corner (exactly 2 neighbors that are not adjacent)
        if len(neighbors) == 2:
            # Check if the two neighbors are not adjacent to each other
            n1, n2 = neighbors
            dr = abs(n1[0] - n2[0])
            dc = abs(n1[1] - n2[1])
            # Neighbors are not adjacent if they're not touching (distance > 1)
            # or if they're diagonal (distance sqrt(2) which is > 1)
            if dr + dc > 1 or (dr == 1 and dc == 1):
                corner_count += 1
    
    # Return a row of corner_count 7s
    return [[7] * corner_count]