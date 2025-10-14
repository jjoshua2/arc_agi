def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    output = [row[:] for row in grid_lst]
    for row in output:
        for i in range(len(row)):
            if row[i] == 7:
                row[i] = 5
    return output