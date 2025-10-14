def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    # Check if all cells in each column are the same
    if all(len(set(col)) == 1 for col in zip(*grid_lst)):
        # It has many columns with the same value, so output a single row with the values of those columns.
        return [[grid_lst[0][c]] for c in range(len(grid_lst[0]))]
    # Check if all cells in each row are the same
    elif all(len(set(row)) == 1 for row in grid_lst):
        # It has many rows with the same value, so output a single column with the values of those rows.
        return [[row[0]] for row in grid_lst]
    else:
        return grid_lst