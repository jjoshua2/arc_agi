from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    q = deque()
    # Initial enqueue for all original 2s: horizontal
    for i in range(rows):
        for j in range(cols):
            if output[i][j] == 2:
                q.append(('h', i, j))
    processed_h = set()
    processed_v = set()
    while q:
        task = q.popleft()
        typ = task[0]
        if typ == 'h':
            r, c = task[1], task[2]
            if (r, c) in processed_h:
                continue
            processed_h.add((r, c))
            # Extend right
            current_c = c
            while True:
                next_c = current_c + 1
                if next_c >= cols:
                    break
                if output[r][next_c] == 0:
                    output[r][next_c] = 2
                    current_c = next_c
                elif output[r][next_c] == 2:
                    current_c = next_c
                else:
                    q.append(('v', current_c, r))
                    break
            # Extend left
            current_c = c
            while True:
                next_c = current_c - 1
                if next_c < 0:
                    break
                if output[r][next_c] == 0:
                    output[r][next_c] = 2
                    current_c = next_c
                elif output[r][next_c] == 2:
                    current_c = next_c
                else:
                    q.append(('v', current_c, r))
                    break
        elif typ == 'v':
            c, r = task[1], task[2]
            if (c, r) in processed_v:
                continue
            processed_v.add((c, r))
            # Extend down
            current_r = r
            while True:
                next_r = current_r + 1
                if next_r >= rows:
                    break
                if output[next_r][c] == 0:
                    output[next_r][c] = 2
                    current_r = next_r
                elif output[next_r][c] == 2:
                    current_r = next_r
                else:
                    q.append(('h', current_r, c))
                    break
            # Extend up
            current_r = r
            while True:
                next_r = current_r - 1
                if next_r < 0:
                    break
                if output[next_r][c] == 0:
                    output[next_r][c] = 2
                    current_r = next_r
                elif output[next_r][c] == 2:
                    current_r = next_r
                else:
                    q.append(('h', current_r, c))
                    break
    return output