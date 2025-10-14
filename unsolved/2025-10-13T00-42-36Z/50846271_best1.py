import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []

    output = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0])

    # Step 1: Gap fill horizontal
    for r in range(rows):
        positions = [c for c in range(cols) if grid[r][c] == 2]
        for j in range(len(positions) - 1):
            p = positions[j]
            q = positions[j + 1]
            for k in range(p + 1, q):
                if grid[r][k] == 5:
                    output[r][k] = 8

    # Step 2: Gap fill vertical
    for c in range(cols):
        positions = [r for r in range(rows) if grid[r][c] == 2]
        for j in range(len(positions) - 1):
            p = positions[j]
            q = positions[j + 1]
            for k in range(p + 1, q):
                if grid[k][c] == 5:
                    output[k][c] = 8

    # Step 3: Find centers
    centers = []
    for i in range(rows):
        for j in range(cols):
            if output[i][j] not in (2, 8):
                continue
            has_horiz = (j > 0 and output[i][j - 1] in (2, 8)) or (j < cols - 1 and output[i][j + 1] in (2, 8))
            has_vert = (i > 0 and output[i - 1][j] in (2, 8)) or (i < rows - 1 and output[i + 1][j] in (2, 8))
            if has_horiz and has_vert:
                centers.append((i, j))

    # Step 4: For each center, extend arms
    for ci, cj in centers:
        # Upper arm length
        upper = 0
        x = ci - 1
        while x >= 0 and output[x][cj] in (2, 8):
            upper += 1
            x -= 1

        # Lower arm length
        lower = 0
        x = ci + 1
        while x < rows and output[x][cj] in (2, 8):
            lower += 1
            x += 1

        # Left arm length
        left = 0
        y = cj - 1
        while y >= 0 and output[ci][y] in (2, 8):
            left += 1
            y -= 1

        # Right arm length
        right = 0
        y = cj + 1
        while y < cols and output[ci][y] in (2, 8):
            right += 1
            y += 1

        max_arm = max(upper, lower, left, right)

        # Extend upper
        diff = max_arm - upper
        if diff > 0:
            ex = ci - 1 - upper
            count = 0
            while count < diff and ex >= 0 and grid[ex][cj] == 5:
                output[ex][cj] = 8
                count += 1
                ex -= 1

        # Extend lower
        diff = max_arm - lower
        if diff > 0:
            ex = ci + 1 + lower
            count = 0
            while count < diff and ex < rows and grid[ex][cj] == 5:
                output[ex][cj] = 8
                count += 1
                ex += 1

        # Extend left
        diff = max_arm - left
        if diff > 0:
            ey = cj - 1 - left
            count = 0
            while count < diff and ey >= 0 and grid[ci][ey] == 5:
                output[ci][ey] = 8
                count += 1
                ey -= 1

        # Extend right
        diff = max_arm - right
        if diff > 0:
            ey = cj + 1 + right
            count = 0
            while count < diff and ey < cols and grid[ci][ey] == 5:
                output[ci][ey] = 8
                count += 1
                ey += 1

    return output