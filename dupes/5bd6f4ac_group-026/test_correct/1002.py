def transform(grid: list[list[int]]) -> list[list[int]]:
    # Extract the top 3 rows and rightmost 3 columns
    return [row[6:9] for row in grid[:3]]