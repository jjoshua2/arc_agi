import numpy as np
from scipy import ndimage

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid_arr = np.array(grid)
    output = grid_arr.copy()
    
    # Find all unique colors (excluding black/0)
    unique_colors = np.unique(grid_arr)
    unique_colors = unique_colors[unique_colors != 0]
    
    for color in unique_colors:
        # Create a binary mask for this color
        mask = (grid_arr == color).astype(int)
        
        # Check if we need to dilate or erode based on the pattern
        # For pink (6) and grey (5), we dilate to fill in
        # For green (3), we erode to hollow out
        
        if color in [5, 6]:  # Grey or pink - dilate
            # Use morphological dilation with a diamond-shaped structuring element
            struct = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
            dilated = ndimage.binary_dilation(mask, structure=struct).astype(int)
            # Set the dilated areas to the original color
            output[dilated == 1] = color
            
        elif color == 3:  # Green - erode
            # Use morphological erosion with a cross-shaped structuring element
            struct = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
            eroded = ndimage.binary_erosion(mask, structure=struct).astype(int)
            # Keep only the eroded parts
            output[grid_arr == color] = 0  # Remove original
            output[eroded == 1] = color    # Add back eroded
    
    return output.tolist()