def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 4 or len(grid[0]) != 19:
        return grid  # Assuming fixed size, but return as is if not
    
    colors = [8, 5, 9, 4]  # A:8, B:5, C:9, D:4
    bases = [0, 5, 10, 15]
    priority = [2, 3, 0, 1]  # C, D, A, B
    
    output = [[0 for _ in range(4)] for _ in range(4)]
    
    for r in range(4):
        for c in range(4):
            for p in priority:
                base = bases[p]
                col = base + c
                if grid[r][col] == colors[p]:
                    output[r][c] = colors[p]
                    break  # Found the highest priority, set and break
    
    return output