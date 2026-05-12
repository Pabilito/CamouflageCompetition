import numpy as np
from scipy import stats

# Each row = position (1st, 2nd, ... 30th)
# Each value = the identity that holds that position in that column
data = np.array([
    [12, 12,  3], [17,  3, 12], [28, 17, 30], [ 3, 23, 16],
    [ 4, 30, 29], [30,  6, 13], [29, 16, 20], [19,  9,  4],
    [24,  4, 17], [23, 10, 28], [16, 18,  5], [ 6,  5, 24],
    [ 5,  1, 18], [10,  8, 27], [ 9, 21, 10], [18, 27, 22],
    [15, 19, 15], [ 8, 28,  9], [22, 24, 19], [14, 25, 21],
    [20, 29, 25], [21, 26, 26], [13, 22, 23], [25, 14,  2],
    [27, 13,  1], [26, 15, 14], [ 1,  7,  8], [11, 20,  6],
    [ 7, 11, 11], [ 2,  2,  7],
])

# Invert: for each identity, find its position (rank) in each column
identities = sorted(set(data.flatten()))

identity_ranks = {}
for ident in identities:
    ranks = []
    for col in range(3):
        pos = np.where(data[:, col] == ident)[0]
        ranks.append(int(pos[0]) + 1)  # 1-based position
    identity_ranks[ident] = ranks

# Build arrays: rank of each identity in A, B, C
rank_A = np.array([identity_ranks[i][0] for i in identities])
rank_B = np.array([identity_ranks[i][1] for i in identities])
rank_C = np.array([identity_ranks[i][2] for i in identities])

cols = {"A": rank_A, "B": rank_B, "C": rank_C}
labels = list(cols.keys())
n = len(labels)

rho_matrix = np.zeros((n, n))
mae_matrix = np.zeros((n, n))

for i, li in enumerate(labels):
    for j, lj in enumerate(labels):
        rho, _ = stats.spearmanr(cols[li], cols[lj])
        rho_matrix[i, j] = rho
        mae_matrix[i, j] = np.mean(np.abs(cols[li] - cols[lj]))

# ── Pretty print ──────────────────────────────────────────────────────────────
SEP = "-" * 52

print("\nIdentity rank lookup (sample: first 10)")
print(SEP)
print(f"  {'ID':>4}  {'rank in A':>10}  {'rank in B':>10}  {'rank in C':>10}")
for ident in identities[:10]:
    ra, rb, rc = identity_ranks[ident]
    print(f"  {ident:>4}  {ra:>10}  {rb:>10}  {rc:>10}")

print(f"\nSpearman Rank Correlation (rho)")
print(SEP)
header = f"{'':>6}" + "".join(f"{lb:>10}" for lb in labels)
print(header)
for i, li in enumerate(labels):
    row = f"{li:>6}" + "".join(f"{rho_matrix[i,j]:>10.4f}" for j in range(n))
    print(row)

print(f"\nMean Absolute Error of Ranks")
print(SEP)
print(header)
for i, li in enumerate(labels):
    row = f"{li:>6}" + "".join(f"{mae_matrix[i,j]:>10.4f}" for j in range(n))
    print(row)

print(f"\nPairwise summary")
print(SEP)
for (li, lj) in [("A","B"), ("A","C"), ("B","C")]:
    i, j = labels.index(li), labels.index(lj)
    print(f"  {li} vs {lj}   rho = {rho_matrix[i,j]:+.4f}   MAE = {mae_matrix[i,j]:.4f}")
