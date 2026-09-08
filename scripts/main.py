# Purpose of script:
# Assess correctness of various RLS implementations on a toy example.
#
# Context:
# Experiments to confirm whether cascaded RLS is indeed a possible replacement
# for standard RLS.
#
# (c) Basil Liekens - ESAT/STADIUS - KU Leuven

import sys

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from cascaded_rls import RLS, cascades


def main():
    rng = np.random.default_rng(seed=42)

    L: int = int(1e4)
    L_h: int = 100
    N: int = 1
    n_levels: int = 1

    u_full = rng.normal(size=(L, L_h))
    h = rng.normal(size=(L_h, N))
    d = u_full @ h

    lmbd = 0.99
    alpha = 0.5

    p_RLS = RLS.RLS_params.construct_params(L_h, N, lmbd)
    p_HRLS = RLS.HRLS_params.construct_params(L_h, N, lmbd)
    p_casc = cascades.cascade_params.construct_params(
        L_h, N, alpha, lmbd, "HRLS", n_levels
    )

    e_RLS = []
    e_HRLS = []
    e_casc = []

    for i in range(L):
        u_k = u_full[[i], :].T
        d_k = d[[i], :]
        w_k = np.ones((*u_k.shape[:-2], 1, 1))

        p_RLS, e_k_RLS = RLS.step_RLS(p_RLS, u_k, d_k, w_k)
        p_HRLS, e_k_HRLS = RLS.step_HRLS(p_HRLS, u_k, d_k, w_k)
        p_casc, e_k_casc = cascades.step_cascaded_RLS(p_casc, u_k, d_k, w_k)

        e_RLS.append(e_k_RLS.item())
        e_HRLS.append(e_k_HRLS.item())
        e_casc.append(e_k_casc.item())

    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(8.5, 5.5)
    ax.semilogy(np.abs(e_RLS), label="RLS")
    ax.semilogy(np.abs(e_HRLS), label="HRLS")
    ax.semilogy(np.abs(e_casc), label="cascaded")
    ax.legend()
    ax.grid(True)
    ax.set(
        xlabel="Iteration index",
        ylabel="Magnitude of error",
        title="Progression of error in toy example",
    )
    ax.autoscale(tight=True, axis="x")

    plt.show(block=True)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    mpl.use("TkAgg")  # avoid issues when plotting
    plt.ion()

    sys.exit(main())
