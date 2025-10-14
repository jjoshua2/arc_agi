import numpy as np
from collections import defaultdict

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Find all connected components (symbols)
    visited = np.zeros_like(grid, dtype=bool)
    symbols = []
    
    def bfs(r, c, color):
        queue = [(r, c)]
        component = []
        while queue:
            cr, cc = queue.pop(0)
            if not (0 <= cr < h and 0 <= cc < w) or visited[cr, cc] or grid[cr, cc] != color:
                continue
            visited[cr, cc] = True
            component.append((cr, cc))
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                queue.append((cr + dr, cc + dc))
        return component
    
    # Find all connected components (ignore background color 0)
    for r in range(h):
        for c in range(w):
            if not visited[r, c] and grid[r, c] != 0:
                color = grid[r, c]
                component = bfs(r, c, color)
                symbols.append((color, component))
    
    # Group symbols by color and count unique symbols
    color_symbols = defaultdict(list)
    for color, component in symbols:
        color_symbols[color].append(component)
    
    # Get unique colors that represent symbols (ignore single cells or very small components)
    # We need to filter out noise - only consider symbols with reasonable size
    unique_colors = []
    for color, components in color_symbols.items():
        # Filter: only consider colors that have at least one component with more than 2 cells
        # or multiple components of the same color
        has_large_component = any(len(comp) > 2 for comp in components)
        has_multiple = len(components) > 1
        if has_large_component or has_multiple:
            unique_colors.append(color)
    
    n = len(unique_colors)
    if n == 0:
        return [[0]]  # No symbols found
    
    # Output grid size: (2n-1) x (2n-1)
    size = 2 * n - 1
    output = np.zeros((size, size), dtype=int)
    
    # Arrange symbols in concentric rings
    # The center gets the first color, then next ring gets next color, etc.
    for i in range(n):
        ring_size = 2 * (n - i) - 1
        start_idx = (size - ring_size) // 2
        end_idx = start_idx + ring_size
        
        # Fill the ring with the i-th color
        color = unique_colors[i]
        output[start_idx:end_idx, start_idx:end_idx] = color
        
        # Remove the inner part to create a ring
        if i < n - 1:
            inner_start = start_idx + 1
            inner_end = end_idx - 1
            output[inner_start:inner_end, inner_start:inner_end] = 0
    
    return output.tolist()