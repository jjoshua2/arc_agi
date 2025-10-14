from collections import deque, Counter
from typing import List, Tuple

def get_neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    
    # Step 1: Find all main components (size >= 3, same color !=0)
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    main_cells = set()  # set of (r, c) that are in main components
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                # BFS to find component size
                color = grid[r][c]
                component = []
                queue = deque([(r, c)])
                visited[r][c] = True
                component.append((r, c))
                
                while queue:
                    cr, cc = queue.popleft()
                    for nr, nc in get_neighbors(cr, cc, rows, cols):
                        if (not visited[nr][nc] and grid[nr][nc] == color):
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            component.append((nr, nc))
                
                if len(component) >= 3:
                    for pr, pc in component:
                        main_cells.add((pr, pc))
    
    # Step 2: Create output
    output = [row[:] for row in grid]
    
    # Step 3: Process non-main non-0 cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in main_cells:
                D = grid[r][c]
                # Get adjacent main cells' colors and positions
                adj_main = []  # list of (color, dir_index) where dir 0:up,1:down,2:left,3:right
                has_up = False
                has_down = False
                has_left = False
                has_right = False
                up_c = None
                down_c = None
                left_c = None
                right_c = None
                
                # up
                if r > 0 and (r-1, c) in main_cells:
                    has_up = True
                    up_c = grid[r-1][c]
                    adj_main.append((up_c, 0))
                
                # down
                if r < rows - 1 and (r+1, c) in main_cells:
                    has_down = True
                    down_c = grid[r+1][c]
                    adj_main.append((down_c, 1))
                
                # left
                if c > 0 and (r, c-1) in main_cells:
                    has_left = True
                    left_c = grid[r][c-1]
                    adj_main.append((left_c, 2))
                
                # right
                if c < cols - 1 and (r, c+1) in main_cells:
                    has_right = True
                    right_c = grid[r][c+1]
                    adj_main.append((right_c, 3))
                
                changed = False
                if D >= 6:
                    # Check horizontal pair
                    if has_left and has_right and left_c == right_c and left_c < 6 and left_c != 0:
                        output[r][c] = left_c
                        changed = True
                    # Check vertical pair
                    elif has_up and has_down and up_c == down_c and up_c < 6 and up_c != 0:
                        output[r][c] = up_c
                        changed = True
                else:
                    # Count for each C <6, !=D
                    count = Counter()
                    for cc, _ in adj_main:
                        if 0 < cc < 6 and cc != D:
                            count[cc] += 1
                    
                    if count:
                        max_cnt = max(count.values())
                        if max_cnt >= 2:
                            candidates = [cc for cc, cnt in count.items() if cnt == max_cnt]
                            # Pick the one with maximum cc in case of tie
                            best_c = max(candidates)
                            output[r][c] = best_c
                            changed = True
                
                if not changed:
                    output[r][c] = 0
    
    return output