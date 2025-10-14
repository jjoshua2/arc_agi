import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    
    # Find the cross color (non-zero color that forms a cross pattern)
    non_zero_colors = np.unique(grid)
    non_zero_colors = non_zero_colors[non_zero_colors != 0]
    
    # If no non-zero colors, return original grid
    if len(non_zero_colors) == 0:
        return grid_lst
    
    # Find which color forms the cross (the one that appears most frequently)
    color_counts = {color: np.sum(grid == color) for color in non_zero_colors}
    cross_color = max(color_counts, key=color_counts.get)
    
    # Find arm color (the other non-zero color)
    arm_color = None
    for color in non_zero_colors:
        if color != cross_color:
            arm_color = color
            break
    
    # If only one color found, use it for both cross and arms
    if arm_color is None:
        arm_color = cross_color
    
    # Create output grid
    output = np.zeros_like(grid)
    
    # Replace cross cells with arm color
    cross_mask = grid == cross_color
    output[cross_mask] = arm_color
    
    # Keep arm cells as they are
    arm_mask = grid == arm_color
    output[arm_mask] = arm_color
    
    return output.tolist()