from typing import List

def find_eight(grid: List[List[int]]) -> tuple[int, int]:
    rows = len(grid)
    for r in range(rows):
        for c in range(len(grid[r])):
            if grid[r][c] == 8:
                return r, c
    return -1, -1  # Not found

def apply_below(grid: List[List[int]], r8: int, c8: int, rows: int, cols: int) -> None:
    col_current = c8
    k = 0
    while True:
        odd_row = r8 + 2 * k + 1
        if odd_row >= rows or col_current < 0:
            break
        # Place single
        if 0 <= col_current < cols:
            grid[odd_row][col_current] = 5
        # Even row
        even_row = r8 + 2 * k + 2
        col_next = col_current - 2
        if even_row < rows and col_next >= 0:
            left = min(col_current, col_next)
            right = max(col_current, col_next)
            for c in range(max(0, left), min(cols, right + 1)):
                grid[even_row][c] = 5
            col_current = col_next
            k += 1
        else:
            # Possible extra single
            if even_row < rows and 0 <= col_current < cols:
                grid[even_row][col_current] = 5
            break

def apply_above(grid: List[List[int]], r8: int, c8: int, rows: int, cols: int) -> None:
    col_current = c8
    k = 0
    while True:
        odd_row = r8 - (2 * k + 1)
        if odd_row < 0 or col_current > cols - 1:
            break
        # Place single
        if 0 <= col_current < cols:
            grid[odd_row][col_current] = 5
        # Even row
        even_row = r8 - (2 * k + 2)
        col_next = col_current + 2
        if even_row >= 0 and col_next <= cols - 1:
            left = min(col_current, col_next)
            right = max(col_current, col_next)
            for c in range(max(0, left), min(cols, right + 1)):
                grid[even_row][c] = 5
            col_current = col_next
            k += 1
        else:
            # Possible extra single
            if even_row >= 0 and 0 <= col_current < cols:
                grid[even_row][col_current] = 5
            break

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    r8, c8 = find_eight(grid)
    if r8 == -1:
        return [row[:] for row in grid]  # No change if no 8
    new_grid = [row[:] for row in grid]
    apply_above(new_grid, r8, c8, rows, cols)
    apply_below(new_grid, r8, c8, rows, cols)
    return new_grid