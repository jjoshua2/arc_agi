from typing import List

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    # Work on copies
    g = [row[:] for row in grid_lst]
    R = len(g)
    C = len(g[0]) if R else 0
    out = [row[:] for row in g]
    
    if R == 0 or C == 0:
        return out

    # Helper: fill all greys between two red 2's in a 1D line (row or column)
    def fill_between_twos_line(values, setter):
        # values: function idx -> value
        # setter: function l..r (exclusive/exclusive) to set 8 for indices between two 2's if all 5
        # Find positions of 2's in order
        idxs = [i for i in range(n_len) if values(i) == 2]
        for k in range(len(idxs) - 1):
            a, b = idxs[k], idxs[k+1]
            if b - a <= 1:
                continue
            # check all in-between are 5
            all_five = True
            for t in range(a + 1, b):
                if values(t) != 5:
                    all_five = False
                    break
            if all_five:
                # set all in-between to 8
                setter(a + 1, b)

    # Step 1: horizontal fill (between two 2's in rows)
    for r in range(R):
        # length for this line
        n_len = C
        def getc(c):
            return g[r][c]
        def set_range(l, r):
            # set out[r][l:r] to 8 where currently 5
            for c in range(l, r):
                if out[r][c] == 5:
                    out[r][c] = 8
        fill_between_twos_line(getc, set_range)

    # Step 2: vertical fill (between two 2's in columns)
    for c in range(C):
        n_len = R
        def getr(r):
            return g[r][c]
        def set_range(l, r):
            for r in range(l, r):
                if out[r][c] == 5:
                    out[r][c] = 8
        fill_between_twos_line(getr, set_range)

    # Step 3: vertical caps
    # Determine which columns should be considered for capping:
    #  - columns where step 2 would have filled (pair of 2's with only 5s in between)
    #  - or columns having any contiguous run of two or more 2's (a vertical bar)
    cols_for_caps = set()

    # from "pair of 2's with 5's between"
    for c in range(C):
        twos = [r for r in range(R) if g[r][c] == 2]
        for k in range(len(twos) - 1):
            r1, r2 = twos[k], twos[k+1]
            if r2 - r1 > 1:
                # check all in-between are 5
                ok = True
                for rr in range(r1 + 1, r2):
                    if g[rr][c] != 5:
                        ok = False
                        break
                if ok:
                    cols_for_caps.add(c)
                    break  # this column qualifies

    # from "contiguous run of 2+ of length >= 2"
    for c in range(C):
        r = 0
        while r < R:
            if g[r][c] == 2:
                start = r
                while r + 1 < R and g[r+1][c] == 2:
                    r += 1
                if r - start + 1 >= 2:
                    cols_for_caps.add(c)
                r += 1
            else:
                r += 1

    # Now, for each qualified column, in the current 'out' grid,
    # find each contiguous block of {2,8} that contains at least two original red 2's
    # and cap one grey (5) just above/below if present.
    for c in cols_for_caps:
        r = 0
        while r < R:
            if out[r][c] in (2, 8):
                start = r
                cnt2 = 0
                while r < R and out[r][c] in (2, 8):
                    if g[r][c] == 2:
                        cnt2 += 1
                    r += 1
                end = r - 1
                if cnt2 >= 2:
                    if start - 1 >= 0 and out[start - 1][c] == 5:
                        out[start - 1][c] = 8
                    if end + 1 < R and out[end + 1][c] == 5:
                        out[end + 1][c] = 8
            else:
                r += 1

    return out