import numpy as np
from collections import Counter

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    eight_positions = np.argwhere(grid == 8)
    if len(eight_positions) == 0:
        return []
    min_r = np.min(eight_positions[:, 0])
    max_r = np.max(eight_positions[:, 0])
    min_c = np.min(eight_positions[:, 1])
    max_c = np.max(eight_positions[:, 1])
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    # Verify the rectangle is all 8s (assuming it is as per examples)
    rect = grid[min_r:max_r+1, min_c:max_c+1]
    if np.all(rect == 8):
        output = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                candidates = []
                for r in range(rows):
                    if r % h == i:
                        for c in range(cols):
                            if c % w == j and grid[r, c] != 8:
                                candidates.append(int(grid[r, c]))
                if candidates:
                    cnt = Counter(candidates)
                    mode = cnt.most_common(1)[0][0]
                    output[i][j] = mode
        return output
    else:
        # If not all 8s, return empty or error, but per problem, assume valid
        return []