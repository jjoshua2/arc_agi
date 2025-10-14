from collections import deque
from typing import List, Tuple, Dict, Set

def transform(grid: List[List[int]]) -> List[List[int]]:
    # Basic checks
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])

    # Copy for output
    out = [row[:] for row in grid]

    # Count color frequencies to find background (most frequent)
    freq: Dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            freq[grid[r][c]] = freq.get(grid[r][c], 0) + 1
    # background color is most frequent color
    background = max(freq.items(), key=lambda x: x[1])[0]

    # Find connected components for non-background colors (4-connected)
    visited = [[False] * cols for _ in range(rows)]
    components = []  # list of dicts: {color, cells, minr,maxr,minc,maxc}
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(rows):
        for c in range(cols):
            if visited[r][c]:
                continue
            color = grid[r][c]
            if color == background:
                visited[r][c] = True
                continue
            # BFS
            q = deque()
            q.append((r,c))
            visited[r][c] = True
            cells = []
            minr, maxr, minc, maxc = r, r, c, c
            while q:
                x,y = q.popleft()
                cells.append((x,y))
                if x < minr: minr = x
                if x > maxr: maxr = x
                if y < minc: minc = y
                if y > maxc: maxc = y
                for dx,dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == color:
                        visited[nx][ny] = True
                        q.append((nx, ny))
            components.append({
                'color': color,
                'cells': cells,
                'minr': minr, 'maxr': maxr,
                'minc': minc, 'maxc': maxc,
                'size': len(cells)
            })

    # Select target components: big components excluding background.
    # Use a threshold to avoid capturing the small motif. Choose a conservative threshold.
    # The motif is tiny (3x3), so threshold = 20 is safe for these tasks.
    threshold = 20
    target_comps = [comp for comp in components if comp['size'] >= threshold]

    # If none found with threshold (fallback to top 1-3 largest non-background comps)
    if not target_comps:
        # choose components with top sizes (but exclude tiny ones)
        comps_sorted = sorted(components, key=lambda x: x['size'], reverse=True)
        # pick up to 4 largest with size >= 5 (if threshold too strict)
        for comp in comps_sorted:
            if comp['size'] >= 5:
                target_comps.append(comp)
            if len(target_comps) >= 4:
                break

    if not target_comps:
        # Nothing to do
        return out

    # Determine candidate center color:
    # Find a color that appears exactly once inside each target component bounding box.
    all_colors: Set[int] = set()
    for r in range(rows):
        for c in range(cols):
            all_colors.add(grid[r][c])
    # Exclude background
    if background in all_colors:
        all_colors.remove(background)

    center_color = None
    num_targets = len(target_comps)
    for col in sorted(all_colors):
        ok = True
        for comp in target_comps:
            cnt = 0
            for rr in range(comp['minr'], comp['maxr'] + 1):
                for cc in range(comp['minc'], comp['maxc'] + 1):
                    if grid[rr][cc] == col:
                        cnt += 1
                        if cnt > 1:
                            break
                if cnt > 1:
                    break
            if cnt != 1:
                ok = False
                break
        if ok:
            center_color = col
            break

    # If not found, fallback to color that appears total num_targets times in grid
    if center_color is None:
        for col in all_colors:
            total = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] == col)
            if total == num_targets:
                center_color = col
                break

    # If still not found, give up (no change)
    if center_color is None:
        return out

    # Find motif center outside target bounding boxes
    def inside_any_target(r,c):
        for comp in target_comps:
            if comp['minr'] <= r <= comp['maxr'] and comp['minc'] <= c <= comp['maxc']:
                return True
        return False

    motif_centers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == center_color and not inside_any_target(r,c):
                motif_centers.append((r,c))
    if not motif_centers:
        # no motif outside (fallback: pick a center from the first target component)
        # use the first component's center coordinate where the center_color is found
        for comp in target_comps:
            found = False
            for rr in range(comp['minr'], comp['maxr']+1):
                for cc in range(comp['minc'], comp['maxc']+1):
                    if grid[rr][cc] == center_color:
                        motif_centers.append((rr,cc))
                        found = True
                        break
                if found:
                    break
        if not motif_centers:
            return out

    motif_r, motif_c = motif_centers[0]

    # Read neighbor colors at the motif center
    def get_color(r,c):
        if 0 <= r < rows and 0 <= c < cols:
            return grid[r][c]
        return None

    U = get_color(motif_r - 1, motif_c)
    D = get_color(motif_r + 1, motif_c)
    L = get_color(motif_r, motif_c - 1)
    R = get_color(motif_r, motif_c + 1)

    # Decide vertical and horizontal colors and whether to do full horizontal extension
    # vertical_color prefer U (or D if U is None)
    vertical_color = None
    if U is not None:
        vertical_color = U
    elif D is not None:
        vertical_color = D

    # horizontal_color choose L (or R)
    horizontal_color = None
    if L is not None:
        horizontal_color = L
    elif R is not None:
        horizontal_color = R

    # Full horizontal extension if all four neighbors exist and are equal
    full_horizontal = False
    if U is not None and D is not None and L is not None and R is not None:
        if (U == D == L == R):
            full_horizontal = True
            horizontal_color = U

    # Apply overlay to each target component
    for comp in target_comps:
        minr, maxr, minc, maxc = comp['minr'], comp['maxr'], comp['minc'], comp['maxc']
        comp_color = comp['color']

        # Find the unique center_color position inside this comp bounding box
        center_pos = None
        for rr in range(minr, maxr + 1):
            for cc in range(minc, maxc + 1):
                if grid[rr][cc] == center_color:
                    center_pos = (rr, cc)
                    break
            if center_pos is not None:
                break
        if center_pos is None:
            # fallback: use bounding-box center
            cr = (minr + maxr) // 2
            cc = (minc + maxc) // 2
            center_pos = (cr, cc)

        cr, cc = center_pos

        # Vertical line: set the column cc across rows inside the rectangle to vertical_color
        if vertical_color is not None:
            for rr in range(minr, maxr + 1):
                # Only overwrite cells that are part of the component (i.e., original comp_color)
                if grid[rr][cc] == comp_color:
                    out[rr][cc] = vertical_color

        # Horizontal:
        if full_horizontal and horizontal_color is not None:
            for ccol in range(minc, maxc + 1):
                if grid[cr][ccol] == comp_color:
                    out[cr][ccol] = horizontal_color
        else:
            # Only color immediate left/right neighbors at center row if they are part of the component
            if cc - 1 >= minc and grid[cr][cc - 1] == comp_color and L is not None:
                out[cr][cc - 1] = L
            if cc + 1 <= maxc and grid[cr][cc + 1] == comp_color and R is not None:
                out[cr][cc + 1] = R

        # Ensure the center remains the center color
        out[cr][cc] = center_color

    return out