from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Vertical straight w=2 downward 2-high
    for r in range(rows - 2):
        for c in range(cols - 1):
            if (grid[r][c] == 5 and grid[r][c + 1] == 5 and
                grid[r + 1][c] == 0 and grid[r + 1][c + 1] == 0 and
                r + 2 < rows and grid[r + 2][c] == 0 and grid[r + 2][c + 1] == 0):
                output[r + 1][c] = 2
                output[r + 1][c + 1] = 2
                output[r + 2][c] = 2
                output[r + 2][c + 1] = 2

    # Vertical L offset right (missing bottom-right) w=2 downward 2-high shifted right
    for r in range(rows - 2):
        for c in range(cols - 2):
            if (grid[r][c] == 5 and grid[r][c + 1] == 5 and
                grid[r + 1][c] == 5 and grid[r + 1][c + 1] == 0 and
                grid[r + 1][c + 2] == 0 and grid[r + 2][c + 1] == 0 and grid[r + 2][c + 2] == 0):
                output[r + 1][c + 1] = 2
                output[r + 1][c + 2] = 2
                output[r + 2][c + 1] = 2
                output[r + 2][c + 2] = 2

    # Vertical L offset left (missing bottom-left) w=2 downward 2-high shifted left
    for r in range(rows - 2):
        for c in range(1, cols - 1):
            if (grid[r][c] == 5 and grid[r][c + 1] == 5 and
                grid[r + 1][c + 1] == 5 and grid[r + 1][c] == 0 and
                grid[r + 1][c - 1] == 0 and grid[r + 2][c] == 0 and grid[r + 2][c - 1] == 0):
                output[r + 1][c] = 2
                output[r + 1][c - 1] = 2
                output[r + 2][c] = 2
                output[r + 2][c - 1] = 2

    # Vertical straight w=3 downward 2-high
    for r in range(rows - 2):
        for c in range(cols - 2):
            if (grid[r][c] == 5 and grid[r][c + 1] == 5 and grid[r][c + 2] == 5 and
                grid[r + 1][c] == 0 and grid[r + 1][c + 1] == 0 and grid[r + 1][c + 2] == 0 and
                r + 2 < rows and grid[r + 2][c] == 0 and grid[r + 2][c + 1] == 0 and grid[r + 2][c + 2] == 0):
                for i in range(3):
                    output[r + 1][c + i] = 2
                    output[r + 2][c + i] = 2

    # Bottom upward multi-high w=2
    if rows >= 2:
        r_bottom = rows - 1
        for c in range(cols - 1):
            if (grid[r_bottom][c] == 0 and grid[r_bottom][c + 1] == 0 and
                grid[r_bottom - 1][c] == 0 and grid[r_bottom - 1][c + 1] == 0 and
                (c == 0 or grid[r_bottom][c - 1] == 5) and
                (c + 2 >= cols or grid[r_bottom][c + 2] == 5)):
                current_r = r_bottom
                while current_r >= 0:
                    if grid[current_r][c] != 0 or grid[current_r][c + 1] != 0:
                        break
                    output[current_r][c] = 2
                    output[current_r][c + 1] = 2
                    current_r -= 1
                    if current_r >= 0:
                        if grid[current_r][c] != 0 or grid[current_r][c + 1] != 0:
                            break

    # Bottom upward multi-high w=3
    if rows >= 2:
        r_bottom = rows - 1
        for c in range(cols - 2):
            if (all(grid[r_bottom][c + i] == 0 for i in range(3)) and
                all(grid[r_bottom - 1][c + i] == 0 for i in range(3)) and
                (c == 0 or grid[r_bottom][c - 1] == 5) and
                (c + 3 >= cols or grid[r_bottom][c + 3] == 5)):
                current_r = r_bottom
                while current_r >= 0:
                    if not all(grid[current_r][c + i] == 0 for i in range(3)):
                        break
                    for i in range(3):
                        output[current_r][c + i] = 2
                    current_r -= 1
                    if current_r >= 0:
                        if not all(grid[current_r][c + i] == 0 for i in range(3)):
                            break

    # Top downward multi-high w=2 (symmetric)
    if rows >= 2:
        r_top = 0
        for c in range(cols - 1):
            if (grid[r_top][c] == 0 and grid[r_top][c + 1] == 0 and
                grid[r_top + 1][c] == 0 and grid[r_top + 1][c + 1] == 0 and
                (c == 0 or grid[r_top][c - 1] == 5) and
                (c + 2 >= cols or grid[r_top][c + 2] == 5)):
                current_r = r_top
                while current_r < rows:
                    if grid[current_r][c] != 0 or grid[current_r][c + 1] != 0:
                        break
                    output[current_r][c] = 2
                    output[current_r][c + 1] = 2
                    current_r += 1
                    if current_r < rows:
                        if grid[current_r][c] != 0 or grid[current_r][c + 1] != 0:
                            break

    # Top downward multi-high w=3 (symmetric)
    if rows >= 2:
        r_top = 0
        for c in range(cols - 2):
            if (all(grid[r_top][c + i] == 0 for i in range(3)) and
                all(grid[r_top + 1][c + i] == 0 for i in range(3)) and
                (c == 0 or grid[r_top][c - 1] == 5) and
                (c + 3 >= cols or grid[r_top][c + 3] == 5)):
                current_r = r_top
                while current_r < rows:
                    if not all(grid[current_r][c + i] == 0 for i in range(3)):
                        break
                    for i in range(3):
                        output[current_r][c + i] = 2
                    current_r += 1
                    if current_r < rows:
                        if not all(grid[current_r][c + i] == 0 for i in range(3)):
                            break

    return output