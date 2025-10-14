import heapq
from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Find all sources: cells with value 1,2,3,4,6,7,8,9? Wait, 9 are targets, so 1-4,6-8
    sources = []
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if 0 < val < 10 and val != 5:
                sources.append((val, r, c))
    # Sort sources by color (small first)
    sources.sort()
    # filled[color] for 0 and 5 cells, initial 10 for unfilled
    filled = [[10 for _ in range(cols)] for _ in range(rows)]
    pq = []
    # Add adjacent to sources to the queue
    for color, r, c in sources:
        for dx, dy in directions:
            nr, nc = r + dx, c + dy
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in (0, 5):
                if filled[nr][nc] > color:
                    filled[nr][nc] = color
                    heapq.heappush(pq, (color, 1, nr, nc))
    # BFS
    while pq:
        color, dist, r, c = heapq.heappop(pq)
        if filled[r][c] < color:
            continue
        # Paint adjacent 9s if still 9
        for dx, dy in directions:
            nr, nc = r + dx, c + dy
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 9 and output[nr][nc] == 9:
                output[nr][nc] = color
        # Propagate to adjacent 0 or 5 if better (smaller color)
        for dx, dy in directions:
            nr, nc = r + dx, c + dy
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in (0, 5) and filled[nr][nc] > color:
                filled[nr][nc] = color
                heapq.heappush(pq, (color, dist + 1, nr, nc))
    # Set remaining 9s to 0
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 9:
                output[r][c] = 0
    return output