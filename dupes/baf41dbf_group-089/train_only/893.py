def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Find green positions (3)
    green_pos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 3]
    if not green_pos:
        return output
    
    min_r = min(r for r, c in green_pos)
    max_r = max(r for r, c in green_pos)
    min_c = min(c for r, c in green_pos)
    max_c = max(c for r, c in green_pos)
    
    # Find pink positions (6)
    pink_pos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 6]
    
    # Compute new bounds
    new_min_c = min_c
    new_max_c = max_c
    new_min_r = min_r
    new_max_r = max_r
    
    for pr, pc in pink_pos:
        if pc < min_c:
            new_min_c = min(new_min_c, pc + 1)
        if pc > max_c:
            new_max_c = max(new_max_c, pc - 1)
        if pr < min_r:
            new_min_r = min(new_min_r, pr + 1)
        if pr > max_r:
            new_max_r = max(new_max_r, pr - 1)
    
    # Identify full and middle rows based on original
    full_rows = []
    middle_rows = []
    for r in range(min_r, max_r + 1):
        green_count = sum(1 for c in range(min_c, max_c + 1) if grid[r][c] == 3)
        if green_count == max_c - min_c + 1:
            full_rows.append(r)
        else:
            middle_rows.append(r)
    
    # Compute original skeleton from middle rows
    if middle_rows:
        skeleton = set(range(min_c, max_c + 1))
        for r in middle_rows:
            row_skel = {c for c in range(min_c, max_c + 1) if grid[r][c] == 3}
            skeleton &= row_skel
        skeleton = sorted(list(skeleton))
        if skeleton:
            left_leg = min(skeleton)
            right_leg = max(skeleton)
            internals = [cc for cc in skeleton if cc != left_leg and cc != right_leg]
        else:
            left_leg = min_c
            right_leg = max_c
            internals = []
    else:
        # If no middle rows, define legs as bounds
        left_leg = min_c
        right_leg = max_c
        internals = []
    
    # Apply horizontal extensions
    if new_min_c < min_c:
        for r in middle_rows:
            if left_leg < cols:
                output[r][left_leg] = 0
            if new_min_c < cols:
                output[r][new_min_c] = 3
        for r in full_rows:
            for cc in range(new_min_c, min_c):
                if cc < cols:
                    output[r][cc] = 3
    
    if new_max_c > max_c:
        for r in middle_rows:
            if right_leg < cols:
                output[r][right_leg] = 0
            if new_max_c < cols:
                output[r][new_max_c] = 3
        for r in full_rows:
            for cc in range(max_c + 1, new_max_c + 1):
                if cc < cols:
                    output[r][cc] = 3
    
    # Now apply vertical extensions
    # Down
    if new_max_r > max_r:
        k = new_max_r - max_r
        # Convert old bottom to sparse
        old_bottom = max_r
        for c in range(new_min_c, new_max_c + 1):
            if c >= cols:
                break
            if c == new_min_c or c == new_max_c or c in internals:
                output[old_bottom][c] = 3
            else:
                output[old_bottom][c] = 0
        # Add (k-1) middle rows
        for i in range(1, k):
            added_r = max_r + i
            if added_r >= rows:
                break
            for c in [new_min_c] + internals + [new_max_c]:
                if c < cols:
                    output[added_r][c] = 3
        # Add new bottom full
        new_bottom = new_max_r
        if new_bottom < rows:
            for c in range(new_min_c, new_max_c + 1):
                if c < cols:
                    output[new_bottom][c] = 3
    
    # Up
    if new_min_r < min_r:
        k = min_r - new_min_r
        # Convert old top to sparse
        old_top = min_r
        for c in range(new_min_c, new_max_c + 1):
            if c >= cols:
                break
            if c == new_min_c or c == new_max_c or c in internals:
                output[old_top][c] = 3
            else:
                output[old_top][c] = 0
        # Add (k-1) middle rows
        for i in range(1, k):
            added_r = min_r - i
            if added_r < 0:
                continue
            for c in [new_min_c] + internals + [new_max_c]:
                if c < cols:
                    output[added_r][c] = 3
        # Add new top full
        new_top = new_min_r
        if new_top >= 0:
            for c in range(new_min_c, new_max_c + 1):
                if c < cols:
                    output[new_top][c] = 3
    
    return output