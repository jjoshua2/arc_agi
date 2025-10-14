from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows, cols = len(grid), len(grid[0])

    # Helper to get 4-connected components for nonzero colors
    visited = [[False]*cols for _ in range(rows)]
    comps = []  # list of dict: {'cells': [(r,c),...], 'color': v, 'r0':..,'c0':..,'r1':..,'c1':..}
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v!=0 and not visited[r][c]:
                q = deque([(r,c)])
                visited[r][c]=True
                cells=[(r,c)]
                while q:
                    x,y=q.popleft()
                    for dx,dy in dirs:
                        nx,ny=x+dx,y+dy
                        if 0<=nx<rows and 0<=ny<cols and not visited[nx][ny] and grid[nx][ny]==v:
                            visited[nx][ny]=True
                            q.append((nx,ny))
                            cells.append((nx,ny))
                r0=min(x for x,_ in cells)
                r1=max(x for x,_ in cells)
                c0=min(y for _,y in cells)
                c1=max(y for _,y in cells)
                comps.append({'cells':cells,'color':v,'r0':r0,'c0':c0,'r1':r1,'c1':c1})

    # Filter to solid squares
    squares=[]
    for comp in comps:
        r0,r1,c0,c1=comp['r0'],comp['r1'],comp['c0'],comp['c1']
        h=r1-r0+1
        w=c1-c0+1
        if h==w and len(comp['cells'])==h*w:
            comp['size']=h
            squares.append(comp)

    # Index squares to detect adjacency quickly
    # We'll just compare all pairs (counts are small)
    def are_adjacent_same_size(a,b):
        S=a['size']
        if b['size']!=S: return None
        # Horizontal adjacency: same rows and touching columns
        if a['r0']==b['r0'] and a['r1']==b['r1']:
            # a left of b
            if a['c1']+1==b['c0']:
                return ('H', 'L')  # a left, b right
            # b left of a
            if b['c1']+1==a['c0']:
                return ('H', 'R')  # b left, a right (we'll orient later)
        # Vertical adjacency: same cols and touching rows
        if a['c0']==b['c0'] and a['c1']==b['c1']:
            # a above b
            if a['r1']+1==b['r0']:
                return ('V', 'T')  # a top, b bottom
            # b above a
            if b['r1']+1==a['r0']:
                return ('V', 'B')  # b top, a bottom
        return None

    # Choose dominant square (higher color value) for each adjacent pair, collect unique expansions
    to_expand=set()  # (r0,c0,S,color)
    for i in range(len(squares)):
        for j in range(i+1,len(squares)):
            a,b=squares[i],squares[j]
            adj=are_adjacent_same_size(a,b)
            if not adj: 
                continue
            # Determine dominant by color value
            if a['color']>=b['color']:
                dom=a
            else:
                dom=b
            to_expand.add( (dom['r0'], dom['c0'], dom['size'], dom['color']) )

    # Draw thick plus for each (only on zeros)
    out=[row[:] for row in grid]
    def paint_rect(r0,r1,c0,c1,color):
        rr0=max(0,r0); rr1=min(rows-1,r1); cc0=max(0,c0); cc1=min(cols-1,c1)
        for r in range(rr0, rr1+1):
            for c in range(cc0, cc1+1):
                if out[r][c]==0:
                    out[r][c]=color

    for r0,c0,S,color in to_expand:
        # central square is already the dominant square; arms of thickness S and length S
        # Up
        paint_rect(r0-S, r0-1, c0, c0+S-1, color)
        # Down
        paint_rect(r0+S, r0+2*S-1, c0, c0+S-1, color)
        # Left
        paint_rect(r0, r0+S-1, c0-S, c0-1, color)
        # Right
        paint_rect(r0, r0+S-1, c0+S, c0+2*S-1, color)

    return out