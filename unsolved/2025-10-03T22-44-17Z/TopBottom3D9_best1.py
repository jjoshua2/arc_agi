from typing import List, Tuple

def find_components(grid: List[List[int]]) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    components = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and (r, c) not in visited:
                component_pos = []
                stack = [(r, c)]
                visited.add((r, c))
                
                while stack:
                    cr, cc = stack.pop()
                    component_pos.append((cr, cc))
                    
                    for dr, dc in directions:
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 3 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                
                if component_pos:
                    max_r = max(pos[0] for pos in component_pos)
                    min_c = min(pos[1] for pos in component_pos)
                    components.append((max_r, min_c, component_pos))
    
    return components

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    
    components = find_components(grid)
    sorted_components = sorted(components, key=lambda x: (x[0], x[1]))
    
    cycle = [1, 6, 2]
    
    output = [row[:] for row in grid]
    
    for i, comp in enumerate(sorted_components):
        color = cycle[i % 3]
        for r, c in comp[2]:
            output[r][c] = color
    
    return output