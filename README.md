# A4Decomposition

A Python implementation of the **A4 rational decomposition** for thermal quantum statistics, designed to efficiently approximate Bose and Fermi functions.

This code decomposes thermal functions into a **small number of exponential terms**, enabling faster and more stable numerical simulations (e.g., for HEOM).

---

## Overview

The central objects decomposed are:

### Bose function
Via the ring-polymer radius of gyration:

$$\mathcal{R}^2(\omega) = \frac{\hbar}{2\omega} \left[ \coth \left( \frac{\beta\hbar\omega}{2} \right) - \frac{2}{\beta\hbar\omega} \right]$$

### Fermi function
Via the 'Fermi pole function':

$$
\mathcal{F}(\omega) = \frac{\hbar}{2\omega}
\tanh\left(\frac{\beta\hbar\omega}{2}\right)
$$

These are both approximated in the form:

$$
F(\omega) = k_0 + \sum_{n=1}^K \frac{k_n}{\omega^2 + \eta_n^2}
$$

where the coefficients $(k_n, \eta_n)$ are obtained via rational approximation.

---

## Features

* Supports **Bose** and **Fermi** thermal distributions.
* **Multiple rational decomposition backends:**
    * `AAA` (adaptive Antoulas–Anderson) **[WORKS BEST]**
    * `ESPRIT_FT` (the ESPRIT algorithm, transformed to time domain) *[LESS GOOD]*
    * `AAA_BT` (balanced truncation variant) *[WORK IN PROGRESS]*
* **Flexible frequency support grids:**
    * Uniform **[WORKS BEST]**
    * Logarithmic
    * Arctanh
    * Quadrature-based
* Automatic pole selection for fixed $K$.
* Optional plotting for diagnostics.

---

## Repository Structure

```text
├── A4Decomposition.py   # Main implementation
├── decompositions.py    # AAA / ESPRIT / AAA_BT routines
├── A4BCF.py             # Bath correlation function [WORK IN PROGRESS]
└── README.md            # Project documentation
```

## Basic Usage

Create an `A4Decomposition` object by specifying the inverse temperature, Planck constant, number of poles, and distribution type. Then call `compute()` to obtain the A4 coefficients.

Then compute the decomposition, and plot if neccessary

```python
from A4Decomposition import A4Decomposition

A4 = A4Decomposition(
    beta=100,
    hbar=1.0,
    K=10,
    distribution="Bose",
    rational_decomposition_type="AAA",
    N_support=10000
)

eta_n, k_n = A4.compute(doplot=True)

