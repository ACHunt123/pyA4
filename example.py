from PyA4 import A4Decomposition

A4 = A4Decomposition(
    beta=100,
    hbar=1.0,
    K=8,
    distribution="Bose",
    rational_decomposition_type="AAA",
    N_support=10000,
    w_max=100
)

eta_n, k_n = A4.compute(doplot=True)
