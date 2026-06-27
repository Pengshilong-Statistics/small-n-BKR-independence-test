# Small-Sample Independence Testing with Blum–Kiefer–Rosenblatt's R

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Shilong Peng**<sup>1</sup>, **Xudong Huang**<sup>2</sup><br>
> <sup>1</sup> School of Statistics and Data Science, Zhejiang Gongshang University, Zhejiang, China<br>
> <sup>2</sup> School of Mathematics and Statistics, Anhui Normal University, Anhui, China

This repository provides the official implementation of the algorithms described in the paper *"Small-Sample Independence Testing with Blum–Kiefer–Rosenblatt's R"*.

## Overview

This project presents **exact small-sample inference methods** for independence testing using the Blum–Kiefer–Rosenblatt (BKR) *R* statistic. By leveraging the Drton–Han–Shi identity that connects *R* with Hoeffding's *D* and Bergsma–Dassios–Yanagimoto's τ\*, we propose efficient computational algorithms for:

1. **Exact null distribution of *R<sub>n</sub>*** for small samples (*n* = 6, …, 11) via exhaustive permutation enumeration — yielding, for the first time, closed-form expressions for the exact null variance.
2. **FFT-based convolution** to compute the approximate null distribution of the aggregated global statistic *Z<sub>Rn</sub>* in high-dimensional settings.

## Repository Structure

```
bkr-exact-inference/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── src/
│   ├── __init__.py              # Package init
│   ├── r_statistic.py           # Algorithm 1: Exact null distribution of R_n
│   └── fft_convolution.py       # Algorithm 2: FFT-based convolution for Z_Rn
└── outputs/                     # Output directory for distribution files
```

## Algorithms

### Algorithm 1 — Exact Null Distribution of *R<sub>n</sub>* (`src/r_statistic.py`)

Computes the **exact finite-sample null distribution** of the BKR *R* statistic by enumerating all *n*! permutations and evaluating the statistic via the Drton–Han–Shi identity:

- Input: sample size *n* (recommended *n* ≤ 10 due to factorial complexity)
- Output: complete probability mass function of *R<sub>n</sub>* (exact fractional values)
- Uses **multiprocessing** for parallel computation across CPU cores
- Produces a distribution file `bkr_distribution_n={n}.txt`

**Key features:**
- Exact arithmetic via Python's `fractions.Fraction`
- Parallelized permutation enumeration with progress bar (`tqdm`)
- Yields exact null moments (mean, variance, etc.)

### Algorithm 2 — FFT-Based Convolution (`src/fft_convolution.py`)

Computes **quantiles of the sum of *N* i.i.d. discrete random variables** using Fast Fourier Transform (FFT) convolution. This is used to approximate the null distribution of the aggregated statistic *Z<sub>Rn</sub>* in high-dimensional testing.

- Input: a discrete distribution (values + probabilities) and convolution count *N*
- Output: specified quantiles of the sum distribution
- Uses `numpy.fft.rfft` / `irfft` for *O*(*N* log *N*) convolution

## Installation

### Requirements

- Python 3.8 or higher
- Dependencies: `numpy`, `tqdm`

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/bkr-exact-inference.git
cd bkr-exact-inference
pip install -r requirements.txt
```

## Quick Start

### Algorithm 1 — Exact *R<sub>n</sub>* Distribution

```python
# Run from command line
python src/r_statistic.py 6
```

This will compute the exact null distribution of *R<sub>7</sub>* (7! = 5,040 permutations) and save results to `outputs/bkr_distribution_n=7.txt`.

**Expected output (example for n=6):**

```
总排列数: 720
使用 8 个 CPU 核心并行计算...
计算 BKR: 100%|████████████| 720/720 [00:01<00:00, 523.14perm/s]

===== 分布摘要 =====
不同 BKR 值的个数: 67

完整分布已保存至: outputs/bkr_distribution_n=6.txt
```

### Algorithm 2 — FFT Quantile Computation

```python
from src.fft_convolution import compute_quantiles

# Define a discrete distribution
values = [-2, -1, 0, 2, 4]
probs  = [128, 320, 64, 128, 80]   # auto-normalized internally
N = 8128                            # convolution count

quantiles = compute_quantiles(values, probs, N, target_probs=(0.025, 0.1, 0.5, 0.9, 0.975))

for p, q in quantiles.items():
    print(f"P(X_sum ≤ {q}) ≈ {p*100:.1f}%")
```

## Citation

If you use this code in your research, please cite our paper.


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration, please contact:

- **Xudong Huang** — huangxdahnu@163.com
