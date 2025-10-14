from collections import deque, Counter
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Function to get components of same color >0 !=8 with size >1
    def get_components(g):
        vis = [[False] * cols for _ in range(rows)]
        components = []
        for i in range(rows):
            for j in range(cols):
                if g[i][j] > 0 and g[i][j] != 8 and not vis[i][j]:
                    color = g[i][j]
                    component = []
                    stack = [(i, j)]
                    vis[i][j] = True
                    while stack:
                        x, y = stack.pop()
                        component.append((x, y))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and not vis[nx][ny] and g[nx][ny] == color:
                                vis[nx][ny] = True
                                stack.append((nx, ny))
                    if len(component) > 1:
                        components.append((color, component))
        return components
    
    components = get_components(grid)
    
    for color, comp in components:
        temp = copy.deepcopy(grid)
        for r, c in comp:
            temp[r][c] = -1
        
        # Flood fill to mark reachable non-wall cells
        vis = [[False] * cols for _ in range(rows)]
        q = deque()
        for r in range(rows):
            for c in range(cols):
                if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and temp[r][c] != -1 and not vis[r][c]:
                    q.append((r, c))
                    vis[r][c] = True
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not vis[nx][ny] and temp[nx][ny] != -1:
                    vis[nx][ny] = True
                    q.append((nx, ny))
        
        # Collect interior colors >0
        color_count = Counter()
        eight_pos = None
        enclosed_pos = []  # all enclosed positions for filling
        for i in range(rows):
            for j in range(cols):
                if not vis[i][j] and temp[i][j] != -1:
                    enclosed_pos.append((i, j))
                    if grid[i][j] > 0:
                        color_count[grid[i][j]] += 1
                        if grid[i][j] == 8:
                            eight_pos = (i, j)  # for single case
        
        if not color_count:
            continue
        
        D = max(color_count, key=color_count.get)
        cnt = color_count[D]
        
        if D == 8 and cnt == 1:
            # Set the single 8 to 0
            if eight_pos:
                output[eight_pos[0]][eight_pos[1]] = 0
        else:
            # Fill all enclosed pos to D
            for pos in enclosed_pos:
                output[pos[0]][pos[1]] = D
    
    # Cleanup: remove single cell components in output
    vis = [[False] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if output[i][j] > 0 and not vis[i][j]:
                colr = output[i][j]
                component = []
                stack = [(i, j)]
                vis[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not vis[nx][ny] and output[nx][ny] == colr:
                            vis[nx][ny] = True
                            stack.append((nx, ny))
                if len(component) == 1:
                    rx, cx = component[0]
                    output[rx][cx] = 0
    
    return output