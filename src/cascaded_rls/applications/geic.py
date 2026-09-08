from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
import scipy as sp

from cascaded_rls import RLS, cascades


@jax.tree_util.register_pytree_node_class
@dataclass
class geic_params:
    """
    Struct containing the state of a "generalized echo and interference canceller"
    (geic) as proposed in [1].

    References
    ----------
    [1] W. Herbordt, W. Kellermann, and S. Nakamura, “Joint optimization of LCMV
        beamforming and acoustic echo cancellation,” in 2005  IEEE Int. Conf. Acoust.
        Speech and Signal Process. (ICASSP), Philadelphia, Pennsylvania, USA, 2005,
        pp. 77–80.
    """

    w_sr: jax.Array
    B: jax.Array
    p_w: cascades.cascade_params | RLS.RLS_params | RLS.HRLS_params

    buff_sc_d: jax.Array
    buff_sc_n: jax.Array
    buff_sc_e: jax.Array

    buff_e: jax.Array

    L: int

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (
            self.w_sr,
            self.B,
            self.p_w,
            self.buff_sc_d,
            self.buff_sc_n,
            self.buff_sc_e,
            self.buff_e,
        )
        aux_data = (self.L,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> Self:
        return cls(*children, *aux_data)

    @classmethod
    def construct_params(
        cls,
        w_sr: jax.Array,
        L: int,
        L_sc: int,
        L_e: int,
        RLS_impl: str,
        lmbd: float,
        n_levels: int = 0,
        batch_shape: tuple = (),
    ) -> Self:
        """
        Construct an initial state struct given the matched filter to use alongside
        the number of loudspeakers and the length of the buffer of both the noise
        referenes and the loudspeaker signals.

        Note:   even though the function and documentation are written assuming
                multidimensional output is allowed, for the filters itself this
                will prove to be somewhat problematic and the constructor should
                still be updated to allow for this.

        Parameters
        ----------
        w_sr : jax.Array, shape `(..., M, 1)`
            The filter vector to use to construct the signal references, `M` is the
            number of microphones.

        L : int, >= 0
            The number of loudspeakers in the setup.

        L_sc : int, >= 0
            The number of noise reference frames to buffer to construct the sidelobe
            cancellation filter.

        L_e : int, >= 0
            The number of loudspeaker signal frames to buffer to construct the echo
            cancellation filters.

        RLS_impl : str
            The RLS implementation to use, supported options are `"RLS"` and `"HRLS"`.

        lmbd : float
            The forgetting factor to use in the RLS updates, should be in (0, 1].

        n_levels : int, 0 or 1, default = 0
            Whether to use an integrated filter (0 levels) or cascaded filter (1 level).

        Returns
        -------
        An instance of `geic_params` with the chosen parameters.
        """
        B = jnp.asarray(sp.linalg.null_space(jnp.conj(w_sr.swapaxes(-2, -1))))

        buff_sc_d = jnp.zeros((*w_sr.shape[:-2], B.shape[-1] * L_sc, 1))
        buff_sc_n = jnp.zeros_like(buff_sc_d)
        buff_sc_e = jnp.zeros_like(buff_sc_d)

        buff_e = jnp.zeros((*w_sr.shape[:-2], L * L_e, 1))

        L_W = buff_e.shape[-2] + buff_sc_d.shape[-2]
        p_e_sr = cascades.cascade_params.construct_params(
            L_W,
            w_sr.shape[-1],
            buff_e.shape[-2] / L_W,
            lmbd,
            RLS_impl=RLS_impl,
            n_levels=n_levels,
            batch_shape=batch_shape,
        )

        return cls(w_sr, B, p_e_sr, buff_sc_d, buff_sc_n, buff_sc_e, buff_e, L)


@jax.jit
def step_geic(
    p: geic_params, m_d_k: jax.Array, m_n_k: jax.Array, m_e_k: jax.Array, e_k: jax.Array
) -> tuple[geic_params, jax.Array, jax.Array, jax.Array]:
    """
    Given a state struct `p`, the contributions of desired signal, noise and echo to the
    microphone signals and the echo signal itself, perform one step of the geic.

    Parameters
    ----------
    p : geic_params
        The state struct containing the current state of the geic.

    m_d_k : jax.Array, shape `(..., M, 1)`
        The contribution of the desired signal to the current microphone signals.

    m_n_k : jax.Array, shape `(..., M, 1)`
        The contribution of the noise to the current microphone signals.

    m_e_k : jax.Array, shape `(..., M, 1)`
        The contribution of the echo signal to the current microphone signals.

    e_k : jax.Array, shape `(..., M, 1)`
        The echo signal at the current timestep.

    Returns
    -------
    An updated state struct alongside the contributions of the desired signal, noise
    and echo to the output signal.

    References
    ----------
    [1] W. Herbordt, W. Kellermann, and S. Nakamura, “Joint optimization of LCMV
        beamforming and acoustic echo cancellation,” in 2005  IEEE Int. Conf. Acoust.
        Speech and Signal Process. (ICASSP), Philadelphia, Pennsylvania, USA, 2005,
        pp. 77–80.
    """
    # Filter signals & update buffers
    sr_d = jnp.conj(p.w_sr.swapaxes(-2, -1)) @ m_d_k
    sr_n = jnp.conj(p.w_sr.swapaxes(-2, -1)) @ m_n_k
    sr_e = jnp.conj(p.w_sr.swapaxes(-2, -1)) @ m_e_k

    nr_d = jnp.conj(p.B.swapaxes(-2, -1)) @ m_d_k
    nr_n = jnp.conj(p.B.swapaxes(-2, -1)) @ m_n_k
    nr_e = jnp.conj(p.B.swapaxes(-2, -1)) @ m_e_k

    n_sc = p.B.shape[-1]
    buff_sc_d = p.buff_sc_d.at[..., n_sc:, :].set(p.buff_sc_d[..., :-n_sc, :])
    buff_sc_d = buff_sc_d.at[..., :n_sc, :].set(nr_d)
    buff_sc_n = p.buff_sc_n.at[..., n_sc:, :].set(p.buff_sc_n[..., :-n_sc, :])
    buff_sc_n = buff_sc_n.at[..., :n_sc, :].set(nr_n)
    buff_sc_e = p.buff_sc_e.at[..., n_sc:, :].set(p.buff_sc_e[..., :-n_sc, :])
    buff_sc_e = buff_sc_e.at[..., :n_sc, :].set(nr_e)

    buff_e = p.buff_e.at[..., p.L :, :].set(p.buff_e[..., : -p.L, :])
    buff_e = buff_e.at[..., : p.L, :].set(e_k)

    ## Compute errors and update filters
    w_k = jnp.ones((*buff_e.shape[:-2], 1, 1))

    u_d_k = jnp.concatenate((jnp.zeros_like(buff_e), buff_sc_d))
    u_n_k = jnp.concatenate((jnp.zeros_like(buff_e), buff_sc_n))
    u_e_k = jnp.concatenate((buff_e, buff_sc_e))

    if not isinstance(p.p_w, cascades.cascade_params):
        p_w, err_d, err_n, err_e = _step_integrated_geic_inner(
            p.p_w, sr_d, sr_n, sr_e, u_d_k, u_n_k, u_e_k, w_k
        )
    else:
        p_w, err_d, err_n, err_e = _step_cascaded_geic_inner(
            p.p_w, sr_d, sr_n, sr_e, u_d_k, u_n_k, u_e_k, w_k
        )

    p_new = geic_params(p.w_sr, p.B, p_w, buff_sc_d, buff_sc_n, buff_sc_e, buff_e, p.L)
    return p_new, err_d, err_n, err_e


def _step_integrated_geic_inner(
    p_w: RLS.RLS_params | RLS.HRLS_params,
    sr_d_k: jax.Array,
    sr_n_k: jax.Array,
    sr_e_k: jax.Array,
    u_d_k: jax.Array,
    u_n_k: jax.Array,
    u_e_k: jax.Array,
    w_k: jax.Array,
) -> tuple[RLS.RLS_params | RLS.HRLS_params, jax.Array, jax.Array, jax.Array]:
    err_d_k = sr_d_k - jnp.conj(p_w.W.swapaxes(-2, -1)) @ u_d_k
    err_n_k = sr_n_k - jnp.conj(p_w.W.swapaxes(-2, -1)) @ u_n_k
    err_e_k = sr_e_k - jnp.conj(p_w.W.swapaxes(-2, -1)) @ u_e_k

    p_w_new, _, _ = p_w.step_function(
        p_w, u_d_k + u_n_k + u_e_k, sr_d_k + sr_n_k + sr_e_k, w_k
    )
    return p_w_new, err_d_k, err_n_k, err_e_k


def _step_cascaded_geic_inner(
    p_w: cascades.cascade_params,
    sr_d_k: jax.Array,
    sr_n_k: jax.Array,
    sr_e_k: jax.Array,
    u_d_k: jax.Array,
    u_n_k: jax.Array,
    u_e_k: jax.Array,
    w_k: jax.Array,
) -> tuple[cascades.cascade_params, jax.Array, jax.Array, jax.Array]:
    d_k_1 = u_d_k[..., : p_w.L_1, :]
    n_k_1 = u_n_k[..., : p_w.L_1, :]
    e_k_1 = u_e_k[..., : p_w.L_1, :]

    d_k_2 = jnp.concatenate((u_d_k[..., p_w.L_1 :, :], sr_d_k), axis=-2)
    n_k_2 = jnp.concatenate((u_n_k[..., p_w.L_1 :, :], sr_n_k), axis=-2)
    e_k_2 = jnp.concatenate((u_e_k[..., p_w.L_1 :, :], sr_e_k), axis=-2)

    d_k_prime = d_k_2 - jnp.conj(p_w.p_1.W.swapaxes(-2, -1)) @ d_k_1  # ty: ignore[unresolved-attribute]
    n_k_prime = n_k_2 - jnp.conj(p_w.p_1.W.swapaxes(-2, -1)) @ n_k_1  # ty: ignore[unresolved-attribute]
    e_k_prime = e_k_2 - jnp.conj(p_w.p_1.W.swapaxes(-2, -1)) @ e_k_1  # ty: ignore[unresolved-attribute]

    err_d_k = (
        d_k_prime[..., p_w.L - p_w.L_1 :, :]
        - jnp.conj(p_w.p_2.W.swapaxes(-2, -1)) @ d_k_prime[..., : p_w.L - p_w.L_1, :]  # ty: ignore[unresolved-attribute]
    )
    err_n_k = (
        n_k_prime[..., p_w.L - p_w.L_1 :, :]
        - jnp.conj(p_w.p_2.W.swapaxes(-2, -1)) @ n_k_prime[..., : p_w.L - p_w.L_1, :]  # ty: ignore[unresolved-attribute]
    )
    err_e_k = (
        e_k_prime[..., p_w.L - p_w.L_1 :, :]
        - jnp.conj(p_w.p_2.W.swapaxes(-2, -1)) @ e_k_prime[..., : p_w.L - p_w.L_1, :]  # ty: ignore[unresolved-attribute]
    )

    p_w_new, _ = p_w.step_function(
        p_w, u_d_k + u_n_k + u_e_k, sr_d_k + sr_n_k + sr_e_k, w_k
    )
    return p_w_new, err_d_k, err_n_k, err_e_k
