"""
Algorithm 1: Exact Null Distribution of Blum–Kiefer–Rosenblatt's R_n

Computes the exact finite-sample null distribution of the BKR R statistic
by enumerating all n! permutations and evaluating the statistic via the
Drton–Han–Shi identity.

Reference:
    Peng, S. & Huang, X. "Small-Sample Independence Testing with
    Blum–Kiefer–Rosenblatt's R" (2025)
"""

import sys
import math
import os
import itertools
from fractions import Fraction
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial


# ---------------------------------------------------------------------------
# Core statistic computation (exact fractional arithmetic)
# ---------------------------------------------------------------------------

def reorder_ranks(x, y):
    """Reorder y-ranks according to sorted x (both 1-based)."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    paired = sorted(zip(x, y), key=lambda t: t[0])
    y_reordered = [t[1] for t in paired]
    sorted_y = sorted(y_reordered)
    rank_map = {val: idx + 1 for idx, val in enumerate(sorted_y)}
    return [rank_map[v] for v in y_reordered]


def hoeffding_d_fraction(x, y):
    """Compute Hoeffding's D (exact fraction) from two sequences of length n."""
    n = len(x)
    if n < 6:
        raise ValueError("Sample size must be at least 6")
    x = list(x)
    y = list(y)

    a = [0] * n
    b = [0] * n
    c = [0] * n

    for i in range(n):
        xi = x[i]
        yi = y[i]
        for j in range(n):
            if x[j] < xi:
                a[i] += 1
            if y[j] < yi:
                b[i] += 1
            if x[j] < xi and y[j] < yi:
                c[i] += 1

    A = sum(a[i] * (a[i] - 1) * b[i] * (b[i] - 1) for i in range(n))
    B = sum((a[i] - 1) * (b[i] - 1) * c[i] for i in range(n))
    C = sum(c[i] * (c[i] - 1) for i in range(n))

    denominator = n * (n - 1) * (n - 2) * (n - 3) * (n - 4)
    numerator = A - 2 * (n - 2) * B + (n - 2) * (n - 3) * C
    return Fraction(numerator, denominator)


def T_fraction(x, y):
    """Compute Bergsma–Dassios–Yanagimoto's tau* (exact fraction)."""
    perm = reorder_ranks(x, y)
    n = len(perm)
    total_concordant = 0
    processed = []

    for i in range(n):
        for j in range(i + 1, n):
            minY = min(perm[i], perm[j])
            maxY = max(perm[i], perm[j])
            num_less = sum(1 for v in processed if v < minY)
            num_greater = sum(1 for v in processed if v > maxY)
            total_concordant += math.comb(num_less, 2) + math.comb(num_greater, 2)
        processed.append(perm[i])

    NC = total_concordant
    denom = n * (n - 1) * (n - 2) * (n - 3)
    x_val = Fraction(NC * 24, denom) - Fraction(1, 3)
    return x_val


def BKR_fraction(x, y, n):
    """Compute BKR R statistic (exact fraction)."""
    D = hoeffding_d_fraction(x, y)
    t = T_fraction(x, y)
    R = t - 12 * D
    return 15 * R * Fraction(math.factorial(n), math.factorial(6))


# ---------------------------------------------------------------------------
# Parallel helper
# ---------------------------------------------------------------------------

def compute_one(p, y, n):
    """Compute BKR value for a single permutation."""
    return BKR_fraction(p, y, n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry: compute exact null distribution of R_n."""
    # Get n from command line or interactive input
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Please provide an integer for n")
            sys.exit(1)
    else:
        n = int(input("Enter n (recommended n ≤ 10): "))

    if n < 5:
        print("Error: n must be at least 5")
        sys.exit(1)

    if n > 10:
        print(f"Warning: n={n} yields {math.factorial(n)} permutations — very large!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            sys.exit(0)

    y = tuple(range(1, n + 1))          # fixed y = 1, 2, ..., n
    total_perms = math.factorial(n)
    print(f"Total permutations: {total_perms}")
    print(f"Using {cpu_count()} CPU cores for parallel computation...")

    # Prepare generator and partial function
    perms_gen = itertools.permutations(range(1, n + 1))
    compute_partial = partial(compute_one, y=y, n=n)

    # Parallel computation and distribution collection
    distribution = {}   # Fraction -> count
    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=total_perms, desc="Computing BKR", unit="perm") as pbar:
            for bkr_val in pool.imap_unordered(compute_partial, perms_gen,
                                                chunksize=1000):
                distribution[bkr_val] = distribution.get(bkr_val, 0) + 1
                pbar.update(1)

    # ---------- Print summary to screen ----------
    print("\n===== Distribution Summary =====")
    print(f"Number of distinct BKR values: {len(distribution)}")
    sorted_by_value = sorted(distribution.items())

    print("BKR values (smallest to largest):")
    if len(sorted_by_value) <= 20:
        for val, cnt in sorted_by_value:
            print(f"  {val} : {cnt} ({cnt/total_perms:.4%})")
    else:
        print("Smallest 10:")
        for val, cnt in sorted_by_value[:10]:
            print(f"  {val} : {cnt} ({cnt/total_perms:.4%})")
        print("  ...")
        print("Largest 10:")
        for val, cnt in sorted_by_value[-10:]:
            print(f"  {val} : {cnt} ({cnt/total_perms:.4%})")

    # ---------- Save full distribution to file ----------
    # Determine output directory (relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    filename = os.path.join(outputs_dir, f"bkr_distribution_n={n}.txt")
    with open(filename, "w") as f:
        f.write(f"# BKR exact null distribution, n = {n}\n")
        f.write(f"# Total permutations = {total_perms}\n")
        f.write("# Format: BKR_fraction  frequency\n")
        for val in sorted(distribution.keys()):
            f.write(f"{val} {distribution[val]}\n")
    print(f"\nFull distribution saved to: {filename}")


if __name__ == "__main__":
    main()
