"""
bkr-exact-inference — Exact Finite-Sample Inference for BKR's R

Algorithms for independence testing using the Blum–Kiefer–Rosenblatt R statistic.

Modules:
    r_statistic     — Algorithm 1: Exact null distribution of R_n
    fft_convolution — Algorithm 2: FFT-based convolution for quantiles
"""

__version__ = "1.0.0"
__author__ = "Shilong Peng, Xudong Huang"

from .r_statistic import BKR_fraction, hoeffding_d_fraction, T_fraction
from .fft_convolution import compute_quantiles

__all__ = [
    "BKR_fraction",
    "hoeffding_d_fraction",
    "T_fraction",
    "compute_quantiles",
]
