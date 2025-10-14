import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = grid.copy()
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Find connected components of 0's, excluding row 0's 0 cells
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and r != 0 and not visited[r, c]:
                component = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    component.append((curr_r, curr_c))
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                            grid[nr, nc] == 0 and nr != 0 and not visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                if len(component) > 0:
                    # Leaf removal on this component
                    current = set(component)
                    while True:
                        leaves = []
                        for pos in list(current):
                            pr, pc = pos
                            neigh_count = 0
                            for dr, dc in directions:
                                nr, nc = pr + dr, pc + dc
                                if (0 <= nr < rows and 0 <= nc < cols and
                                    (nr, nc) in current):
                                    neigh_count += 1
                            if neigh_count == 1:
                                leaves.append(pos)
                        
                        if not leaves:
                            break
                        
                        for pos in leaves:
                            current.discard(pos)
                    
                    # Fill remaining with 2
                    for pr, pc in current:
                        output[pr, pc] = 2

    return output.tolist()