def transform(grid: list[list[int]]) -> list[list[int]]:
    out = []
    for row in grid:
        rev = row[::-1]
        out_row = row + rev + row + rev + row
        out.append(out_row)
    return out