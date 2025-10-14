def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows_num = len(grid_lst)
    if rows_num == 0:
        return []
    cols_num = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    for s in range(rows_num):
        for e in range(s + 1, rows_num):
            # compute mask
            mask = [True] * cols_num
            for col in range(cols_num):
                for r in range(s, e + 1):
                    if grid_lst[r][col] != 0:
                        mask[col] = False
                        break
            
            # find runs
            col = 0
            while col < cols_num:
                if not mask[col]:
                    col += 1
                    continue
                start = col
                while col < cols_num and mask[col]:
                    col += 1
                end = col - 1
                length = end - start + 1
                if length >= 3:
                    for r in range(s, e + 1):
                        for c in range(start, end + 1):
                            output[r][c] = 6
    return output