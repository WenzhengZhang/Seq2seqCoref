import numpy as np

match = 1
mismatch = -1
gap = -1
gap_start = -1
gap_ext = 0
# gap_ext = 0.00001
# gap_ext = 0.01
# gap_ext = 0.1
# gap_ext = 0.001


def match_score(a, b):
    if a == b:
        return match
    else:
        return mismatch


def max_track(m, x, y, mode):
    max_value = max(m, x, y)
    if mode == 'h':
        if max_value == x:
            max_type = 2
        elif max_value == m:
            max_type = 1
        elif max_value == y:
            max_type = 3
    elif mode == 'l':
        if max_value == y:
            max_type = 3
        elif max_value == m:
            max_type = 1
        elif max_value == x:
            max_type = 2
    else:
        raise ValueError('wrong alignment mode')
    return max_value, max_type


def global_align(input_ids, rec_ids):
    cost = np.zeros((len(input_ids) + 1, len(
        rec_ids) + 1))  # cost of alignment between tokens[:i]
    # and output_tokens[:j]
    best = np.zeros_like(cost,
                         dtype=int)  # best choice when aligning tokens[:i] and output_tokens[:j]

    for i in range(len(input_ids) + 1):
        for j in range(len(rec_ids) + 1):
            if i == 0 and j == 0:
                continue

            candidates = []

            # match
            if i > 0 and j > 0:
                candidates.append(
                    ((0 if input_ids[i - 1] == rec_ids[
                        j - 1] else 1) + cost[i - 1, j - 1], 1))

            # skip in the first sequence
            if i > 0:
                candidates.append((1 + cost[i - 1, j], 2))

            # skip in the second sequence
            if j > 0:
                candidates.append((1 + cost[i, j - 1], 3))

            chosen_cost, chosen_option = min(candidates)
            cost[i, j] = chosen_cost
            best[i, j] = chosen_option

    # reconstruct best alignment
    matching = {}

    i = len(input_ids) - 1
    j = len(rec_ids) - 1

    while i >= 0 and j >= 0:
        chosen_option = best[i + 1, j + 1]

        if chosen_option == 1:
            # match
            matching[j] = i
            i, j = i - 1, j - 1

        elif chosen_option == 2:
            # skip in the first sequence
            i -= 1

        else:
            # skip in the second sequence
            j -= 1
    return matching


def affine_global_align(x, y, pad_token, mode):
    """Global alignment with affine penalties. We assume we are maximizing."""
    M = np.zeros((len(x) + 1, len(y) + 1), dtype=float)
    X = np.zeros((len(x) + 1, len(y) + 1), dtype=float)
    Y = np.zeros((len(x) + 1, len(y) + 1), dtype=float)
    # from M,X,Y
    # keep track last position as well as last alignment type
    # 1: M, 2: X, 3: Y
    track_M = np.zeros((len(x) + 1, len(y) + 1, 3), dtype=int)
    track_X = np.zeros((len(x) + 1, len(y) + 1, 3), dtype=int)
    track_Y = np.zeros((len(x) + 1, len(y) + 1, 3), dtype=int)
    # initialize
    M[0, 0] = 0
    for i in range(1, len(x) + 1):
        M[i][0] = -float('inf')
        X[i][0] = gap_start + i * gap_ext
        Y[i][0] = -float('inf')
        track_X[i, 0, 0] = 2
        track_X[i, 0, 1] = i - 1
        track_X[i, 0, 2] = 0

    for i in range(1, len(y) + 1):
        M[0][i] = -float('inf')
        X[0][i] = -float('inf')
        Y[0][i] = gap_start + i * gap_ext
        track_Y[0, i, 0] = 3
        track_Y[0, i, 1] = 0
        track_Y[0, i, 2] = i - 1

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            M_max_value, M_max_type = max_track(
                M[i - 1][j - 1],
                X[i - 1][j - 1],
                Y[i - 1][j - 1], mode
            )
            M[i][j] = match_score(x[i - 1], y[j - 1]) + M_max_value
            track_M[i, j, 0] = M_max_type
            track_M[i, j, 1] = i - 1
            track_M[i, j, 2] = j - 1

            X_max_value, X_max_type = max_track(
                gap_start + gap_ext + M[i - 1][j],
                gap_ext + X[i - 1][j],
                gap_start + gap_ext + Y[i - 1][j], mode
            )
            X[i, j] = X_max_value
            track_X[i, j, 0] = X_max_type
            track_X[i, j, 1] = i - 1
            track_X[i, j, 2] = j

            Y_max_value, Y_max_type = max_track(
                gap_start + gap_ext + M[i][j - 1],
                gap_start + gap_ext + X[i][j - 1],
                gap_ext + Y[i][j - 1], mode
            )

            Y[i][j] = Y_max_value
            track_Y[i, j, 0] = Y_max_type
            track_Y[i, j, 1] = i
            track_Y[i, j, 2] = j - 1
    # traceback here
    max_i, max_j = len(x), len(y)
    x_aligned, y_aligned = [], []
    # x_aligned, y_aligned ="",""
    opt, track_type = max_track(
        M[max_i, max_j], X[max_i, max_j], Y[max_i, max_j], mode
    )
    matching = {}
    while max_i > 0 or max_j > 0:
        if track_type == 1:
            x_aligned.append(x[max_i - 1])
            y_aligned.append(y[max_j - 1])
            # x_aligned += x[max_i - 1]
            # y_aligned += y[max_j - 1]
            track_mat = track_M
            matching[max_j - 1] = max_i - 1
        elif track_type == 2:
            x_aligned.append(x[max_i - 1])
            y_aligned.append(pad_token)
            # x_aligned += x[max_i - 1]
            # y_aligned += '-'
            track_mat = track_X
        elif track_type == 3:
            x_aligned.append(pad_token)
            y_aligned.append(y[max_j - 1])
            # x_aligned += '-'
            # y_aligned += y[max_j - 1]
            track_mat = track_Y
        else:
            raise ValueError('wrong track type')
        track_type = track_mat[max_i, max_j, 0]
        max_i = track_mat[max_i, max_j, 1]
        max_j = track_mat[max_i, max_j, 2]

    return x_aligned[::-1], y_aligned[::-1], matching
