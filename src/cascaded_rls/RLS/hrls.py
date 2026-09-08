from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass
class HRLS_params:
    """
    Struct containing all data required for implementing the Householder RLS
    algorithm presented in [1].

    References
    ----------
    [1] J. Wung et al., “Robust Multichannel Linear Prediction for Online
        Speech Dereverberation Using Weighted Householder Least Squares Lattice
        Adaptive Filter,” IEEE Trans. Signal Process., vol. 68, pp. 3559–3574, 2020,
        doi: 10.1109/TSP.2020.2997201.
    """

    S_n: jax.Array
    W: jax.Array
    lmbd_inv_sqrt: float
    step_function: Callable

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.S_n, self.W)
        aux_data = (self.lmbd_inv_sqrt, self.step_function)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> Self:
        return cls(*children, *aux_data)

    @classmethod
    def construct_params(
        cls, L: int, N: int, lmbd: float, batch_shape: tuple = ()
    ) -> Self:
        return cls._construct_params(L, N, lmbd, batch_shape, step_HRLS)

    @classmethod
    def _construct_params(
        cls, L: int, N: int, lmbd: float, batch_shape: tuple, step_function: Callable
    ) -> Self:
        S_n = jnp.tile(jnp.eye(L), reps=batch_shape + (1, 1))
        W = jnp.zeros(batch_shape + (L, N))

        return cls(S_n, W, (1 / jnp.sqrt(lmbd).item()), step_function)


def _step_HRLS(
    p: HRLS_params, x_k: jax.Array, d_k: jax.Array, w_k: jax.Array
) -> tuple[HRLS_params, jax.Array, jax.Array]:
    """
    Given filter state `p`, input vector `u_k` and desired signal `d_k`,
    perform one step of the HRLS algorithm [1]. No assumption on the structure
    of the input is made, hence the full input vector should be provided at
    all times.

    Parameters
    ----------
    p: HRLS_params
        The current state vector containing square root factor `S_n`, shape
        `(..., L, L)`, filter weights `(..., L, N)` and forgetting factor
        `lmbd`.

    x_k: jax.Array, `(..., L, 1)`
        Input vector at current timestep. Should be the complete input vector
        to account for cases where the input is not transversal.

    d_k: jax.Array, `(..., N, 1)`
        Desired signal vector.

    w_k: jax.Array, `(..., 1, 1)`
        Possibly time-varying weight to apply at the current timestep.

    Returns
    -------
    A tuple containing
    -   The updated state struct
    -   The (a priori) error signal, shape `(..., N, 1)`
    -   The denominator of the Kalman gain, zeta_k, could be required to pass
        to the next stage in a cascaded implementation.

    References
    ----------
    [1] J. Wung et al., “Robust Multichannel Linear Prediction for Online
        Speech Dereverberation Using Weighted Householder Least Squares Lattice
        Adaptive Filter,” IEEE Trans. Signal Process., vol. 68, pp. 3559–3574, 2020,
        doi: 10.1109/TSP.2020.2997201.
    """

    xi_k = d_k - jnp.conj(p.W.swapaxes(-2, -1)) @ x_k

    S_n_tilde = p.lmbd_inv_sqrt * p.S_n

    u_k = jnp.conj(S_n_tilde.swapaxes(-2, -1)) @ x_k
    zeta_k = w_k + jnp.conj(u_k.swapaxes(-2, -1)) @ u_k
    d_k = jnp.sqrt(zeta_k)
    k_k = S_n_tilde @ (u_k / zeta_k)

    S_k = S_n_tilde - d_k / (jnp.sqrt(w_k) + d_k) * k_k @ jnp.conj(u_k.swapaxes(-2, -1))
    W_k = p.W + k_k @ jnp.conj(xi_k.swapaxes(-2, -1))

    p_k = HRLS_params(S_k, W_k, p.lmbd_inv_sqrt, p.step_function)
    return p_k, xi_k, zeta_k


@jax.jit
def step_HRLS(
    p: HRLS_params, x_k: jax.Array, d_k: jax.Array, w_k: jax.Array
) -> tuple[HRLS_params, jax.Array]:
    p_k, e_k, _ = _step_HRLS(p, x_k, d_k, w_k)
    return p_k, e_k
