def transform(grid: list[list[int]]) -> list[list[int]]:
    tl, tr = grid[0]
    bl, br = grid[1]
    row0 = [tl, tr] * 3
    row1 = [bl, br] * 3
    row2 = [tr, tl] * 3
    row3 = [br, bl] * 3
    return [
        row0,
        row1,
        row2,
        row3,
        row0,
        row1
    ]