import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = [row[:] for row in grid_lst]
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    for r1 in range(rows):
        for r2 in range(r1, rows):
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    # Check if all interior == 0
                    all_zero = True
                    for r in range(r1, r2 + 1):
                        for c in range(c1, c2 + 1):
                            if grid[r][c] != 0:
                                all_zero = False
                                break
                        if not all_zero:
                            break
                    if not all_zero:
                        continue
                    # Check borders
                    valid = True
                    # Top
                    if r1 > 0:
                        for c in range(c1, c2 + 1):
                            if grid[r1 - 1][c] != 5:
                                valid = False
                                break
                    if not valid:
                        continue
                    # Bottom
                    if r2 < rows - 1:
                        for c in range(c1, c2 + 1):
                            if grid[r2 + 1][c] != 5:
                                valid = False
                                break
                    if not valid:
                        continue
                    # Left
                    if c1 > 0:
                        for r in range(r1, r2 + 1):
                            if grid[r][c1 - 1] != 5:
                                valid = False
                                break
                    if not valid:
                        continue
                    # Right
                    if c2 < cols - 1:
                        for r in range(r1, r2 + 1):
                            if grid[r][c2 + 1] != 5:
                                valid = False
                                break
                    if not valid:
                        continue
                    # Fill with 4
                    for r in range(r1, r2 + 1):
                        for c in range(c1, c2 + 1):
                            grid[r][c] = 4
    return grid