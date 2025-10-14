def transform(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)
    # Deep copy the grid
    output = [row[:] for row in grid]
    
    # Find positions on the main diagonal where value is 1
    pos = [i for i in range(n) if grid[i][i] == 1]
    
    if len(pos) < 2:
        return output
    
    # Sort positions (though they should be in order)
    pos.sort()
    
    # Compute common difference d
    d = pos[1] - pos[0]
    
    # Assume all differences are the same; for the puzzle, they are
    
    # Last position with 1
    last = pos[-1]
    
    # Continue the sequence
    next_pos = last + d
    while next_pos < n:
        output[next_pos][next_pos] = 2
        next_pos += d
    
    return output