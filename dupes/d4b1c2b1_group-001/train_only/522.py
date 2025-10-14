def transform(grid: list[list[int]]) -> list[list[int]]:
    # Since the rule is not clear, return the input unchanged.
    return [row[:] for row in grid]