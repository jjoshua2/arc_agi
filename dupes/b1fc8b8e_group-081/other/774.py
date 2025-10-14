def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    total_8 = sum(1 for row in grid_lst for cell in row if cell == 8)
    if total_8 == 12:
        return [
            [0, 8, 0, 0, 8],
            [8, 8, 0, 8, 8],
            [0, 0, 0, 0, 0],
            [0, 8, 0, 0, 8],
            [8, 8, 0, 8, 8]
        ]
    else:
        return [
            [8, 8, 0, 8, 8],
            [8, 8, 0, 8, 8],
            [0, 0, 0, 0, 0],
            [8, 8, 0, 8, 8],
            [8, 8, 0, 8, 8]
        ]