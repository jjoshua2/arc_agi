def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    h = len(grid_lst)
    w = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    for i in range(h):
        for j in range(i + 1, h):
            height = j - i + 1
            if height < 2:
                continue
            for k in range(w):
                for l in range(k + 1, w):
                    width = l - k + 1
                    if width < 2:
                        continue
                    # Check if all cells in [i..j][k..l] are 0
                    all_zero = True
                    for r in range(i, j + 1):
                        for c in range(k, l + 1):
                            if grid_lst[r][c] != 0:
                                all_zero = False
                                break
                        if not all_zero:
                            break
                    if all_zero:
                        # Fill with 2
                        for r in range(i, j + 1):
                            for c in range(k, l + 1):
                                output[r][c] = 2
    return output