"""
Algorithm 2: FFT-Based Convolution for Quantile Computation

Computes quantiles of the sum of N i.i.d. discrete random variables using
Fast Fourier Transform (FFT) convolution.

Used to approximate the null distribution of the aggregated global statistic
Z_Rn in high-dimensional independence testing.

Reference:
    Peng, S. & Huang, X. "Small-Sample Independence Testing with
    Blum–Kiefer–Rosenblatt's R" (2025)
"""

import numpy as np
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Low-level core algorithms
# ---------------------------------------------------------------------------

def build_single_step_pmf(values, probs):
    """
    Build a single-step probability mass function (PMF), mapping arbitrary
    integer values to a dense array.

    Parameters
    ----------
    values : array-like
        All possible integer values of the random variable X.
    probs : array-like
        Corresponding probabilities (auto-normalized internally).

    Returns
    -------
    pmf : np.ndarray (float32)
        Dense PMF array.
    min_v : int
        Minimum value.
    max_v : int
        Maximum value.
    """
    values = np.asarray(values, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    probs /= probs.sum()  # auto-normalize

    min_v, max_v = values.min(), values.max()
    span = max_v - min_v + 1
    offset = -min_v

    pmf = np.zeros(span, dtype=np.float32)
    for v, p in zip(values, probs):
        pmf[v + offset] += p

    return pmf, min_v, max_v


def compute_sum_pmf_via_fft(pmf, N):
    """
    Compute the PMF of the sum of N i.i.d. random variables via FFT.

    Parameters
    ----------
    pmf : np.ndarray
        Single-step PMF.
    N : int
        Number of convolutions (i.i.d. copies).

    Returns
    -------
    pmf_sum : np.ndarray (float64)
        PMF of the sum distribution.
    out_span : int
        Length of the output PMF array.
    """
    span = len(pmf)
    out_span = int(N * span - (N - 1))
    fft_len = 1 << (out_span - 1).bit_length()

    P = np.fft.rfft(pmf, fft_len)
    P_N = P ** N
    pmf_sum = np.fft.irfft(P_N, fft_len)[:out_span].astype(np.float64)
    pmf_sum /= pmf_sum.sum()  # numerical normalization

    return pmf_sum, out_span


def compute_quantiles_from_cdf(pmf_sum, min_v, max_v, N, target_probs):
    """
    Compute target quantiles from the CDF of the sum distribution.

    Parameters
    ----------
    pmf_sum : np.ndarray
        PMF of the sum distribution.
    min_v, max_v : int
        Min/max values of the single-step distribution.
    N : int
        Number of convolutions.
    target_probs : tuple of float
        Target cumulative probabilities (e.g., (0.025, 0.975)).

    Returns
    -------
    OrderedDict : {probability: quantile (int)}
    """
    cdf = np.cumsum(pmf_sum)
    z_values = np.arange(N * min_v, N * max_v + 1, dtype=np.int64)

    results = OrderedDict()
    for p in sorted(target_probs):
        idx = int(np.searchsorted(cdf, p, side='left'))
        if idx < len(z_values):
            results[p] = int(z_values[idx])
        else:
            results[p] = None
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_quantiles(values, probs, N, target_probs=(0.025, 0.975)):
    """
    Compute specified quantiles of the sum of N i.i.d. discrete random variables.

    Parameters
    ----------
    values : list or array
        All possible values of the random variable X (must be integers).
    probs : list or array
        Corresponding probabilities (auto-normalized; frequency weights OK).
    N : int
        Number of i.i.d. copies to sum (convolution count).
    target_probs : tuple of float
        Target cumulative probabilities. Default: (0.025, 0.975).

    Returns
    -------
    OrderedDict
        Keys are target probabilities; values are the corresponding
        integer quantiles. Example: {0.025: 10, 0.975: 30}

    Raises
    ------
    ValueError
        If values and probs have different lengths.
    TypeError
        If values are not all integers.
    """
    # 1. Input validation
    if len(values) != len(probs):
        raise ValueError("values and probs must have the same length!")
    if not all(isinstance(v, (int, np.integer)) for v in values):
        raise TypeError(
            "All values must be integers. "
            "If using decimals, first multiply by a common denominator."
        )

    # 2. Build single-step distribution
    pmf, min_v, max_v = build_single_step_pmf(values, probs)

    # 3. FFT-based N-fold convolution
    pmf_sum, out_span = compute_sum_pmf_via_fft(pmf, N)

    # 4. Compute CDF and extract quantiles
    results = compute_quantiles_from_cdf(pmf_sum, min_v, max_v, N, target_probs)

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: single-step distribution
    values = [-2, -1, 0, 2, 4]
    probs = [128, 320, 64, 128, 80]  # auto-normalized
    N = 8128

    import time

    t0 = time.time()
    quantiles = compute_quantiles(
        values, probs, N,
        target_probs=(0.025, 0.1, 0.5, 0.9, 0.975)
    )
    elapsed = time.time() - t0

    print(f"Computed {N} convolutions in {elapsed:.4f} seconds")
    print("\nQuantile results:")
    for p, q in quantiles.items():
        print(f"  P(X_sum ≤ {q}) ≈ {p * 100:.1f}%")
