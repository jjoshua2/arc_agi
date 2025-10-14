from typing import Dict, List, Tuple

def find_components(grid: List[List[int]]) -> List[Dict]:
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                color = grid[i][j]
                comp: List[Tuple[int, int]] = []
                stack: List[Tuple[int, int]] = [(i, j)]
                visited[i][j] = True
                comp.append((i, j))
                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                            comp.append((nx, ny))
                if comp:
                    min_r = min(r for r, _ in comp)
                    max_r = max(r for r, _ in comp)
                    min_c = min(c for _, c in comp)
                    max_c = max(c for _, c in comp)
                    height = max_r - min_r + 1
                    width = max_c - min_c + 1
                    qualifies_extra = (
                        min_r == 0
                        and max_r == 3
                        and width >= 4
                        and all(grid[0][k] == color for k in range(min_c, max_c + 1))
                        and all(grid[3][k] == color for k in range(min_c, max_c + 1))
                    )
                    pos = 0 if min_c < 3 else 1 if min_c < 6 else 2
                    components.append(
                        {
                            "pos": pos,
                            "color": color,
                            "min_r": min_r,
                            "max_r": max_r,
                            "min_c": min_c,
                            "max_c": max_c,
                            "height": height,
                            "width": width,
                            "qualifies_extra": qualifies_extra,
                            "size": len(comp),
                            "cells": comp,
                        }
                    )
    return components


def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    components = find_components(grid_lst)
    # Group by pos, choose largest if multiple
    pos_comps = {0: None, 1: None, 2: None}
    for comp in components:
        p = comp["pos"]
        if pos_comps[p] is None or comp["size"] > pos_comps[p]["size"]:
            pos_comps[p] = comp
    output_rows = []
    for p in range(3):
        comp = pos_comps[p]
        added_something = False
        if comp is not None:
            c = comp["color"]
            minc = comp["min_c"]
            w = comp["width"]
            h = comp["height"]
            qualifies = comp["qualifies_extra"]
            if qualifies:
                l = 3
            else:
                if p == 0 or p == 2:
                    if w >= 4:
                        l = 3
                    elif h == 4:
                        l = 2
                    else:
                        l = 1
                else:  # p == 1
                    if minc == 4:
                        if w == 3 and h == 3:
                            l = 1
                        else:
                            l = 3
                    else:  # minc == 5
                        if w == 4 and h == 4:
                            l = 3
                        else:
                            l = 2
            row = [c] * l + [0] * (3 - l)
            output_rows.append(row)
            added_something = True
            if qualifies and len(output_rows) < 3:
                if p == 0:
                    e = 1
                else:
                    e = 3
                extra_row = [c] * e + [0] * (3 - e)
                output_rows.append(extra_row)
                added_something = True
        if not added_something and len(output_rows) < 3:
            output_rows.append([0, 0, 0])
    # Ensure exactly 3 rows, pad if less (though shouldn't happen)
    while len(output_rows) < 3:
        output_rows.append([0, 0, 0])
    # Truncate if more (though shouldn't)
    return output_rows[:3]