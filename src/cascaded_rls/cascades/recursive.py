from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp

from cascaded_rls.RLS.hrls import HRLS_params, _step_HRLS
from cascaded_rls.RLS.rls import RLS_params, _step_RLS


@jax.tree_util.register_pytree_node_class
@dataclass
class cascade_params:
    p_1: cascade_params | RLS_params | HRLS_params  # parameters to create u_1' and d'
    p_2: cascade_params | RLS_params | HRLS_params  # second filtering stage
    buff_2d: jax.Array  # Buffer: avoid repeated allocations to concatenate u2 and d

    L: int
    L_1: int
    step_function: Callable

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.p_1, self.p_2, self.buff_2d)
        aux_data = (self.L, self.L_1, self.step_function)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> Self:
        return cls(*children, *aux_data)

    @classmethod
    def construct_params(
        cls,
        L: int,
        N: int,
        alpha: float,
        lmbd: float,
        RLS_impl: str,
        n_levels: int,
        level: int = 0,
        batch_shape: tuple = (),
    ) -> Self | RLS_params | HRLS_params:
        """
        Given the required attributes, construct an instance of `cascade_params`.
        Specific things requiring attention:
        - `alpha`:      the split between the length of the first segment and the
                        second one, i.e. L_1 = round(alpha * L).
        - `RLS_impl`:   which RLS algorithm to use, `RLS` and `HRLS` are supported.
        - `n_levels`:   how deep to recursively split the filters, i.e. there are
                        2^n_levels smaller subfilters. Guards are in place to ensure
                        the length of the filter stays sensible.
        - `level`:      the current level to keep track of how deep to recurse, should
                        in most cases only be used internally.
        """
        match RLS_impl:
            case "RLS":
                step_function = _step_RLS
                params = RLS_params
            case "HRLS":
                step_function = _step_HRLS
                params = HRLS_params
            case _:
                raise ValueError(f"Unsupported RLS algorithm received, got {RLS_impl}")

        L_1 = round(alpha * L)
        L_2 = L - L_1
        buff_2d = jnp.zeros(batch_shape + (L_2 + N, 1))

        # Ensure a split to always lead to both parts having at least 1 element
        if n_levels == level or L_1 < 1 or L - L_1 < 1:
            return params._construct_params(L, N, lmbd, batch_shape, step_function)

        else:
            p_1 = cascade_params.construct_params(
                L_1, N + L_2, alpha, lmbd, RLS_impl, n_levels, level + 1, batch_shape
            )
            p_2 = cascade_params.construct_params(
                L_2, N, alpha, lmbd, RLS_impl, n_levels, level + 1, batch_shape
            )
            return cls(p_1, p_2, buff_2d, L, L_1, step_cascaded_RLS)


@jax.jit
def step_cascaded_RLS(
    p: cascade_params | RLS_params | HRLS_params,
    u_k: jax.Array,
    d_k: jax.Array,
    w_k: jax.Array,
) -> tuple[cascade_params | RLS_params | HRLS_params, jax.Array]:
    p_k, e_k, _ = _step_cascaded_RLS(p, u_k, d_k, w_k)
    return p_k, e_k


def _step_cascaded_RLS(
    p: cascade_params | RLS_params | HRLS_params,
    u_k: jax.Array,
    d_k: jax.Array,
    w_k: jax.Array,
) -> tuple[cascade_params | RLS_params | HRLS_params, jax.Array, jax.Array]:
    """
    Given the current filter state `p`, input vector `u_k` and desired signal `d_k`,
    perform one step of the cascaded RLS algorithm and return updated parameters
    alongside the error signal. It is expected the entire input vector is passed in.

    Parameters
    ----------
    p : cascade_params
        The struct containing the filter state, at the "deepest level" this becomes
        either one of `RLS_params` or `HRLS_params` to stop the recursion.

    x_k: jax.Array, `(..., L, 1)`
        Input vector for the current step.

    d_k: jax.Array, `(..., N, 1)`
        Desired signal vector.

    w_k: jax.Array, `(..., 1, 1)`
        Possibly time-varying weight to apply at the current timestep.

    Returns
    -------
    A tuple containing
        -   The updated struct
        -   The (a priori) error signal, shape `(..., N, 1)`
        -   The denominator of the Kalman gain, zeta_k, could be required to pass
            to the next stage of a cascaded implementation.
    """
    if not isinstance(p, cascade_params):
        return p.step_function(p, u_k, d_k, jnp.asarray(w_k))

    else:
        u_k_1 = u_k[..., : p.L_1, :]
        u_k_2 = u_k[..., p.L_1 :, :]

        buff_2d = p.buff_2d.at[..., : p.L - p.L_1, :].set(u_k_2)
        buff_2d = buff_2d.at[..., p.L - p.L_1 :, :].set(d_k)

        p_1_up, e_prime, zeta_1 = _step_cascaded_RLS(p.p_1, u_k_1, buff_2d, w_k)

        u_2_prime = e_prime[..., : p.L - p.L_1, :]
        d_prime = e_prime[..., p.L - p.L_1 :, :]
        p_2_up, e_k, zeta_2 = _step_cascaded_RLS(p.p_2, u_2_prime, d_prime, zeta_1)

        p_up = cascade_params(p_1_up, p_2_up, buff_2d, p.L, p.L_1, p.step_function)
        return p_up, e_k, zeta_2
