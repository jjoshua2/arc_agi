def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    # Find the largest rectangle with uniform perimeter
    # by iterating over possible top, bottom, left, right
    # and check that all cells in the top and bottom rows and left and right columns have the same value.
    # Then, extract the interior: rows from i+1 to j-1, columns from l+1 to r-1.
    # But then, the output would be a grid of size (j-i-1) x (r-l-1)
    # which is not the same size as the input.
    return grid.tolist()