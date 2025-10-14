from collections import defaultdict
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all initial 1 positions
    row_to_cols: defaultdict[int, List[int]] = defaultdict(list)
    col_to_rows: defaultdict[int, List[int]] = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                row_to_cols[r].append(c)
                col_to_rows[c].append(r)
    
    # Compute line positions
    line_positions: set[Tuple[int, int]] = set()
    for r, cls in row_to_cols.items():
        if len(cls) >= 2:
            min_c = min(cls)
            max_c = max(cls)
            for cc in range(min_c, max_c + 1):
                line_positions.add((r, cc))
    
    for c, rws in col_to_rows.items():
        if len(rws) >= 2:
            min_r = min(rws)
            max_r = max(rws)
            for rr in range(min_r, max_r + 1):
                line_positions.add((rr, c))
    
    # Find connected components of 2's
    def get_red_components() -> List[List[Tuple[int, int]]]:
        visited = [[False] * cols for _ in range(rows)]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        components = []
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2 and not visited[i][j]:
                    component = []
                    stack = [(i, j)]
                    visited[i][j] = True
                    while stack:
                        x, y = stack.pop()
                        component.append((x, y))
                        for dx, dy in directions:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 2:
                                visited[nx][ny] = True
                                stack.append((nx, ny))
                    if component:
                        components.append(component)
        return components
    
    components = get_red_components()
    
    # Find components that intersect lines
    to_fill: set[Tuple[int, int]] = set()
    for comp in components:
        if any((rr, cc) in line_positions for rr, cc in comp):
            for pos in comp:
                to_fill.add(pos)
    
    # Create output grid
    output = [row[:] for row in grid]
    
    # Set line positions to 1
    for r, c in line_positions:
        output[r][c] = 1
    
    # Fill the components
    for r, c in to_fill:
        output[r][c] = 1
    
    return output