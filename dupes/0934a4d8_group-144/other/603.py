def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or len(grid_lst) != 13 or len(grid_lst[0]) != 4:
        return []  # Assuming standard input size
    output = [[0 for _ in range(4)] for _ in range(6)]
    for r in range(6):
        for c in range(4):
            top_val = grid_lst[r][c]
            bot_val = grid_lst[r + 7][c]
            if bot_val == 6:
                output[r][c] = 0
            else:
                output[r][c] = 8 if top_val == 0 else 0
    return output