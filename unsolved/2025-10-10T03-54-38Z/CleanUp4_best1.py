from collections import Counter

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    h = len(grid)
    w = len(grid[0])

    # Count colors and pick the most frequent as background
    counts = Counter()
    for r in range(h):
        for c in range(w):
            counts[grid[r][c]] += 1
    background, _ = counts.most_common(1)[0]

    # Initialize output filled with background color
    output = [[background for _ in range(w)] for _ in range(h)]

    # Collect positions per color
    positions_by_color = {}
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val == background:
                continue
            positions_by_color.setdefault(val, set()).add((r, c))

    # For each color (except background), detect rectangles via corners
    rectangles = []  # list of (top, bottom, left, right, color)
    for val, posset in positions_by_color.items():
        pts = list(posset)
        n = len(pts)
        found = set()
        # Check all pairs of distinct points to be opposite corners
        for i in range(n):
            r1, c1 = pts[i]
            for j in range(i + 1, n):
                r2, c2 = pts[j]
                # need different rows and columns to form a rectangle
                if r1 == r2 or c1 == c2:
                    continue
                top = min(r1, r2)
                bottom = max(r1, r2)
                left = min(c1, c2)
                right = max(c1, c2)
                # only consider rectangles with interior (at least 3x3)
                if bottom - top < 2 or right - left < 2:
                    continue
                # check that the other two corner positions are present
                if (top, right) in posset and (bottom, left) in posset:
                    key = (top, bottom, left, right)
                    if key not in found:
                        found.add(key)
                        rectangles.append((top, bottom, left, right, val))
        # done for this color

    # Draw each rectangle's border into the output
    for top, bottom, left, right, val in rectangles:
        # top and bottom rows
        for col in range(left, right + 1):
            output[top][col] = val
            output[bottom][col] = val
        # left and right columns
        for row in range(top, bottom + 1):
            output[row][left] = val
            output[row][right] = val

    return output