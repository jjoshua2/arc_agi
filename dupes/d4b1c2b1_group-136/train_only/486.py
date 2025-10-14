def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Find line color by checking full rows and columns
    line_color = None
    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]) and grid[r, 0] != 0:
            line_color = grid[r, 0]
            break
    if line_color is None:
        for c in range(w):
            if np.all(grid[:, c] == grid[0, c]) and grid[0, c] != 0:
                line_color = grid[0, c]
                break
    if line_color is None:
        return grid.tolist()
    
    # Find horizontal dividing lines (full rows of line_color)
    h_dividers = []
    for r in range(h):
        if np.all(grid[r, :] == line_color):
            h_dividers.append(r)
    
    # Find vertical dividing lines (full columns of line_color)
    v_dividers = []
    for c in range(w):
        if np.all(grid[:, c] == line_color):
            v_dividers.append(c)
    
    # Create regions based on dividers
    h_regions = []
    prev = -1
    for d in h_dividers:
        if d > prev + 1:
            h_regions.append((prev + 1, d - 1))
        prev = d
    if prev < h - 1:
            h_regions.append((prev + 1, h - 1))
    
    v_regions = []
    prev = -1
    for d in v_dividers:
        if d > prev + 1:
            v_regions.append((prev + 1, d - 1))
        prev = d
    if prev < w - 1:
            v_regions.append((prev + 1, w - 1))
    
    # Create output grid
    output = grid.copy()
    
    # For each pair of diagonally opposite regions, mirror the colored patterns
    num_h_regions = len(h_regions)
    num_v_regions = len(v_regions)
    
    for i in range(num_h_regions):
        for j in range(num_v_regions):
            # Get the symmetric region (diagonally opposite)
            sym_i = num_h_regions - 1 - i
            sym_j = num_v_regions - 1 - j
            
            if i == sym_i and j == sym_j:
                continue  # Skip the center region if it exists
                
            # Source region
            h_start, h_end = h_regions[i]
            v_start, v_end = v_regions[j]
            source_region = grid[h_start:h_end+1, v_start:v_end+1]
            
            # Target region
            sym_h_start, sym_h_end = h_regions[sym_i]
            sym_v_start, sym_v_end = v_regions[sym_j]
            target_region = output[sym_h_start:sym_h_end+1, sym_v_start:sym_v_end+1]
            
            # Check if regions have the same size
            if source_region.shape != target_region.shape:
                continue
                
            # Mirror the colored patterns (non-zero and non-line_color) from source to target
            for r in range(source_region.shape[0]):
                for c in range(source_region.shape[1]):
                    source_val = source_region[r, c]
                    if source_val != 0 and source_val != line_color:
                        # Mirror position
                        mirror_r = source_region.shape[0] - 1 - r
                        mirror_c = source_region.shape[1] - 1 - c
                        target_region[mirror_r, mirror_c] = source_val
    
    return output.tolist()