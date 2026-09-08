from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass
class RLS_params:
    R_inv: jax.Array
    W: jax.Array
    lmbd: float
    step_function: Callable

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.R_inv, self.W)
        aux_data = (self.lmbd, self.step_function)

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> Self:
        return cls(*children, *aux_data)

    @classmethod
    def construct_params(
        cls, L: int, N: int, lmbd: float, batch_shape: tuple = ()
    ) -> Self:
        """
        Given the dimensions `L` and `N` and the forgetting factor `lmbd`,
        initialize an instance of the parameters using an identity for `R_inv`.
        """
        return cls._construct_params(L, N, lmbd, batch_shape, step_RLS)

    @classmethod
    def _construct_params(
        cls, L: int, N: int, lmbd: float, batch_shape: tuple, step_function: Callable
    ) -> Self:
        R_inv = jnp.tile(jnp.eye(L), reps=batch_shape + (1, 1))
        W = jnp.zeros(batch_shape + (L, N))

        return cls(R_inv, W, lmbd, step_function)


def _step_RLS(
    p: RLS_params, x_k: jax.Array, d_k: jax.Array, w_k: jax.Array
) -> tuple[RLS_params, jax.Array, jax.Array]:
    """
    Given the state `p`, input vector `u_k` and desired signal `d_k`, perform
    one step of the RLS algorithm and return updated parameters alongside the
    error signal. No explicit assumption about the input structure is made,
    hence the entire input vector should be passed in at any time.

    Parameters
    ----------
    p: RLS_params
        The struct containing the state of the RLS algorithm. Should contain
        the inverse covariance matrix `R`, which has shape `(..., L, L)`, the
        filter `W` of shape `(..., L, N)` and the forgetting factor `lmbd`.

    x_k: jax.Array, `(..., L, 1)`
        Input vector for the current step.

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
    """
    e_k = d_k - jnp.conj(p.W.swapaxes(-2, -1)) @ x_k

    # Compute Kalman gain `k_k`
    R_inv_tilde = p.R_inv / p.lmbd
    v_k = R_inv_tilde @ x_k
    zeta_k = w_k + jnp.conj(x_k.swapaxes(-2, -1)) @ v_k
    k_k = v_k / zeta_k

    # Update (inverse) covariance and filter
    R_inv_k = R_inv_tilde - k_k @ jnp.conj(v_k.swapaxes(-2, -1))
    W_k = p.W + k_k @ jnp.conj(e_k.swapaxes(-2, -1))

    p_k = RLS_params(R_inv_k, W_k, p.lmbd, p.step_function)
    return p_k, e_k, zeta_k


@jax.jit
def step_RLS(
    p: RLS_params, x_k: jax.Array, d_k: jax.Array, w_k: jax.Array
) -> tuple[RLS_params, jax.Array]:
    p_k, e_k, _ = _step_RLS(p, x_k, d_k, w_k)
    return p_k, e_k
