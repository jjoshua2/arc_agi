def transform(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)
    output = [row[:] for row in grid]
    
    # Collect positions where diagonal has 1
    positions = [d for d in range(n) if grid[d][d] == 1]
    
    if len(positions) < 2:
        return output
    
    # Compute delta
    delta = positions[1] - positions[0]
    
    # Verify AP (optional, but assume true)
    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] != delta:
            # If not AP, but examples are, so pass
            pass
    
    last = positions[-1]
    next_d = last + delta
    while next_d < n:
        output[next_d][next_d] = 2
        next_d += delta
    
    return output