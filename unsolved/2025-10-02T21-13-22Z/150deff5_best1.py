def transform(grid: list[list[int]]) -> list[list[int]]:
    output = [row[:] for row in grid]
    for i in range(len(output)):
        for j in range(len(output[i])):
            if output[i][j] == 5:
                output[i][j] = 8
            elif output[i][j] == 8:
                output[i][j] = 5
    return output