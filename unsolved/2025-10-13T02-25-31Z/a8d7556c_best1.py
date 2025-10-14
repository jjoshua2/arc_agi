import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = grid.copy()
    for top in range(rows):
        for bottom in range(top, rows):
            is_full = [True] * cols
            for c in range(cols):
                for r in range(top, bottom + 1):
                    if grid[r, c] != 0:
                        is_full[c] = False
                        break
            c = 0
            while c < cols:
                if not is_full[c]:
                    c += 1
                    continue
                left = c
                c += 1
                while c < cols and is_full[c]:
                    c += 1
                right = c - 1
                width = right - left + 1
                height = bottom - top + 1
                can_up = top > 0 and all(grid[top - 1, cc] == 0 for cc in range(left, right + 1))
                can_down = bottom < rows - 1 and all(grid[bottom + 1, cc] == 0 for cc in range(left, right + 1))
                if not can_up and not can_down:
                    if (width == 2 and height >= 2) or (height == 2 and width >= 2):
                        for r in range(top, bottom + 1):
                            for cc in range(left, right + 1):
                                output[r, cc] = 2
    return output.tolist()