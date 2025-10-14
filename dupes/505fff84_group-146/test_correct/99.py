def transform(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    
    segments = []
    for r in range(h):
        ones = [c for c in range(w) if grid[r][c] == 1]
        eights = [c for c in range(w) if grid[r][c] == 8]
        if len(ones) == 1 and len(eights) == 1:
            left = ones[0]
            right = eights[0]
            if left < right:
                segments.append((r, left, right))
    
    if not segments:
        return []
    
    segments.sort(key=lambda x: x[0])
    
    # Assume all have same width
    first_width = segments[0][2] - segments[0][1] - 1
    output = []
    for r, left, right in segments:
        seg_width = right - left - 1
        if seg_width != first_width:
            # In problem, they match, but to handle, perhaps take min or something, but assume match
            pass
        row_out = []
        for c in range(left + 1, right):
            val = grid[r][c]
            row_out.append(2 if val == 2 else 0)
        output.append(row_out)
    
    return output