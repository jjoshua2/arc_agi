import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    left_column = grid[:, 0]
    top_row = grid[0, :]
    is_columns = np.all(left_column == left_column[0])
    if is_columns:
        out_in_order = list(dict.fromkeys(top_row))
        return [out_in_order]  # as a list of lists (row)
    else:
        out_in_order = list(dict.fromkeys(left_column))
        return [[x] for x in out_in_order]  # as a list of lists (column)