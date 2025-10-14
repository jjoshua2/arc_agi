import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find green bar columns (columns where all cells are 3)
    green_bars = []
    for c in range(cols):
        if np.all(grid[:, c] == 3):
            green_bars.append(c)
    
    # If no green bars found, return original grid
    if not green_bars:
        return grid.tolist()
    
    # Create output grid (copy of input)
    output = grid.copy()
    
    # Process each section between green bars
    sections = []
    for i in range(len(green_bars) - 1):
        start_col = green_bars[i] + 1
        end_col = green_bars[i + 1] - 1
        if start_col <= end_col:  # Valid section
            sections.append((start_col, end_col))
    
    # Also process section before first bar and after last bar
    if green_bars[0] > 0:
        sections.append((0, green_bars[0] - 1))
    if green_bars[-1] < cols - 1:
        sections.append((green_bars[-1] + 1, cols - 1))
    
    # For each section, collect yellow cells and move them
    for start_col, end_col in sections:
        section_width = end_col - start_col + 1
        if section_width <= 0:
            continue
            
        # Find all yellow cells in this section
        yellow_positions = []
        for r in range(rows):
            for c in range(start_col, end_col + 1):
                if grid[r, c] == 4:
                    yellow_positions.append((r, c))
        
        # Clear all yellow cells from this section in output
        for r in range(rows):
            for c in range(start_col, end_col + 1):
                if output[r, c] == 4:
                    output[r, c] = 0
        
        # Place yellow cells at positions 1 cell from right edge
        target_col = end_col - 1  # 1 cell from right edge
        if target_col >= start_col:  # Valid target position
            for r, c in yellow_positions:
                output[r, target_col] = 4
    
    return output.tolist()