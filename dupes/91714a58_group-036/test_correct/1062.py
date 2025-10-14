def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    if rows == 0:
        return []
    cols = len(grid_lst[0])
    
    max_area = 0
    best_i, best_j, best_k, best_l = -1, -1, -1, -1
    best_c = 0
    
    for i in range(rows):
        for j in range(i, rows):
            for k in range(cols):
                for l in range(k, cols):
                    # Check if all cells in [i..j, k..l] are the same c > 0
                    is_uniform = True
                    first_c = -1
                    for r in range(i, j + 1):
                        for cc in range(k, l + 1):
                            val = grid_lst[r][cc]
                            if first_c == -1:
                                first_c = val
                            elif val != first_c:
                                is_uniform = False
                                break
                        if not is_uniform:
                            break
                    if is_uniform and first_c > 0:
                        area = (j - i + 1) * (l - k + 1)
                        if area > max_area:
                            max_area = area
                            best_i, best_j, best_k, best_l = i, j, k, l
                            best_c = first_c
    
    # Create output: all zeros except the best rectangle
    output = [[0] * cols for _ in range(rows)]
    if max_area > 0:
        for r in range(best_i, best_j + 1):
            for cc in range(best_k, best_l + 1):
                output[r][cc] = best_c
    return output