from collections import Counter

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 3 or len(grid[0]) != 3:
        raise ValueError("Input must be a 3x3 grid")
    
    # Flatten to count frequencies
    flat = [cell for row in grid for cell in row]
    counts = Counter(flat)
    
    if not counts:
        # All zero or empty, return 9x9 zeros
        return [[0] * 9 for _ in range(9)]
    
    min_freq = min(counts.values())
    # Get colors with minimal frequency, pick the smallest color if tie
    candidate_colors = [col for col, cnt in counts.items() if cnt == min_freq]
    C = min(candidate_colors)
    
    # Find positions where grid[r][c] == C
    positions = []
    for r in range(3):
        for c in range(3):
            if grid[r][c] == C:
                positions.append((r, c))
    
    # Create 9x9 output initialized to 0
    output = [[0] * 9 for _ in range(9)]
    
    # For each position, copy the entire input grid to the corresponding 3x3 block
    for pos_r, pos_c in positions:
        for i in range(3):
            for j in range(3):
                output[3 * pos_r + i][3 * pos_c + j] = grid[i][j]
    
    return output