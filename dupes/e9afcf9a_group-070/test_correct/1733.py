def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 2:
        return [row[:] for row in grid]
    
    output = []
    for i in range(len(grid)):
        output.append([])
        for j in range(len(grid[0])):
            if j % 2 == 0:
                # Even columns: keep original pattern
                output[i].append(grid[i][j])
            else:
                # Odd columns: swap rows
                output[i].append(grid[1 - i][j])
    
    return output