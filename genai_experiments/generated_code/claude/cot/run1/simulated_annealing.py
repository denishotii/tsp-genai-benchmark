import numpy as np
import random
import math

def solve(dist_matrix):
    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(D[0, 1] + D[1, 0])

    def tour_length(t):
        return float(D[t, np.roll(t, -1)].sum())

    # Nearest-neighbor initial tour
    def nn_tour(start=0):
        visited = np.zeros(n, dtype=bool)
        t = np.empty(n, dtype=int)
        t[0] = start
        visited[start] = True
        for k in range(1, n):
            last = t[k - 1]
            d = D[last].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            t[k] = nxt
            visited[nxt] = True
        return t

    best_tour = nn_tour(0)
    best_len = tour_length(best_tour)

    # Try a few NN starts
    for s in range(1, min(n, 5)):
        t = nn_tour(s)
        L = tour_length(t)
        if L < best_len:
            best_len = L
            best_tour = t

    current = best_tour.copy()
    current_len = best_len

    # Estimate initial temperature from sample deltas
    sample_deltas = []
    for _ in range(min(200, n * n)):
        i = random.randint(1, n - 1)
        j = random.randint(1, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        a = current[i - 1]; b = current[i]
        c = current[j]; d = current[(j + 1) % n]
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        if delta > 0:
            sample_deltas.append(delta)
    if sample_deltas:
        avg = sum(sample_deltas) / len(sample_deltas)
        T = avg / math.log(2.0)  # ~50% acceptance for average worsening
    else:
        T = 1.0
    T = max(T, 1e-12)

    alpha = 0.9995
    T_min = 1e-8

    # Iteration budget scales with n
    max_iters = max(20000, 200 * n * n)
    max_iters = min(max_iters, 2_000_000)

    no_improve_limit = max(5000, 50 * n)
    no_improve = 0

    for it in range(max_iters):
        # Pick i < j with i >= 1 (avoid full reversal)
        i = random.randint(1, n - 1)
        j = random.randint(1, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        a = current[i - 1]
        b = current[i]
        c = current[j]
        d = current[(j + 1) % n]
        # If (j+1)%n == i-1 (only when j==n-1 and i==0, already excluded) skip
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]

        if delta < 0 or random.random() < math.exp(-delta / T):
            current[i:j + 1] = current[i:j + 1][::-1]
            current_len += delta
            if current_len < best_len - 1e-12:
                best_len = current_len
                best_tour = current.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T *= alpha
        if T < T_min:
            T = T_min

        if no_improve >= no_improve_limit and T <= T_min * 10:
            break

    # Final 2-opt local polish
    improved = True
    t = best_tour.copy()
    L = best_len
    passes = 0
    while improved and passes < 50:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            a = t[i - 1]; b = t[i]
            dab = D[a, b]
            for j in range(i + 1, n):
                c = t[j]; d = t[(j + 1) % n]
                if (j + 1) % n == i - 1:
                    continue
                delta = D[a, c] + D[b, d] - dab - D[c, d]
                if delta < -1e-12:
                    t[i:j + 1] = t[i:j + 1][::-1]
                    L += delta
                    improved = True
                    b = t[i]
                    dab = D[a, b]
        if L < best_len:
            best_len = L
            best_tour = t.copy()

    return best_tour.tolist(), float(best_len)