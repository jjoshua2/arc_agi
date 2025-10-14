def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    max_area = 0
    best_i, best_j, best_k, best_l, best_c = -1, -1, -1, -1, 0
    
    for i in range(rows):
        for j in range(i, rows):
            for k in range(cols):
                for l in range(k, cols):
                    c = grid_lst[i][k]
                    if c == 0:
                        continue
                    all_same = True
                    for r in range(i, j + 1):
                        for cl in range(k, l + 1):
                            if grid_lst[r][cl] != c:
                                all_same = False
                                break
                        if not all_same:
                            break
                    if all_same:
                        area = (j - i + 1) * (l - k + 1)
                        if area > max_area:
                            max_area = area
                            best_i, best_j, best_k, best_l, best_c = i, j, k, l, c
    
    # Create output grid all zeros
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Fill the best rectangle
    if max_area > 0:
        for r in range(best_i, best_j + 1):
            for cl in range(best_k, best_l + 1):
                output[r][cl] = best_c
    
    return output