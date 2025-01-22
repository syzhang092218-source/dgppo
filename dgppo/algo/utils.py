import jax
import jax.numpy as jnp
import einops as ei

from jaxtyping import Float

from ..utils.typing import Array, TFloat
from ..utils.utils import assert_shape


def compute_dec_ocp_gae(
    Tah_hs: Float[Array, "T a nh"],
    T_l: TFloat,
    Tp1ah_Vh: Float[Array, "Tp1 a nh"],
    Tp1_Vl: Float[Array, "Tp1"],
    disc_gamma: float,
    gae_lambda: float,
    discount_to_max: bool = True
) -> tuple[Float[Array, "T a nh"], TFloat]:
    """
    Compute GAE for MASOCP. Compute it using DP, starting at V(x_T) and working backwards.

    Returns
    -------
    Qhs: (T, a, nh),
    Ql: (T,)
    """
    T, n_agent, nh = Tah_hs.shape

    def loop(carry, inp):
        ii, hs, l, Vhs, Vl = inp  # hs: (a, nh), Vhs: (a, nh)
        next_Vhs_row, next_Vl_row, gae_coeffs = carry

        mask = assert_shape(jnp.arange(T + 1) < ii + 1, T + 1)
        mask_l = assert_shape(mask[:, None], (T + 1, 1))
        mask_h = assert_shape(mask[:, None, None], (T + 1, 1, 1))

        # DP for Vh.
        if discount_to_max:
            h_disc = hs.max(-1)  # (a,)
        else:
            h_disc = hs

        disc_to_h = (1 - disc_gamma) * h_disc[None, :, None] + disc_gamma * next_Vhs_row  # (T + 1, a, h)
        Vhs_row = assert_shape(mask_h * jnp.maximum(hs, disc_to_h), (T + 1, n_agent, nh), "Vhs_row")
        # DP for Vl. Clamp it to within J_max so it doesn't get out of hand.
        Vl_row = assert_shape(mask_l * (l + disc_gamma * next_Vl_row), (T + 1, n_agent))
        cat_V_row = assert_shape(jnp.concatenate([Vhs_row, Vl_row[:, :, None]], axis=-1), (T + 1, n_agent, nh + 1))

        Qs_GAE = assert_shape(ei.einsum(cat_V_row, gae_coeffs, "Tp1 na nhp2, Tp1 -> na nhp2"), (n_agent, nh + 1))

        # Setup Vs_row for next timestep.
        Vhs_row = Vhs_row.at[ii + 1, :].set(Vhs)
        Vl_row = Vl_row.at[ii + 1].set(Vl)

        #                            *  *        *   *             *     *
        # Update GAE coeffs. [1] -> [λ 1-λ] -> [λ² λ(1-λ) 1-λ] -> [λ³ λ²(1-λ) λ(1-λ) 1-λ]
        gae_coeffs = jnp.roll(gae_coeffs, 1)
        gae_coeffs = gae_coeffs.at[0].set(gae_lambda ** (ii + 1))
        gae_coeffs = gae_coeffs.at[1].set((gae_lambda ** ii) * (1 - gae_lambda))

        return (Vhs_row, Vl_row, gae_coeffs), Qs_GAE

    init_gae_coeffs = jnp.zeros(T + 1)
    init_gae_coeffs = init_gae_coeffs.at[0].set(1.0)

    Tah_Vh, T_Vl = Tp1ah_Vh[:-1], Tp1_Vl[:-1][:, None].repeat(n_agent, axis=1)
    Vh_final, Vl_final = Tp1ah_Vh[-1], Tp1_Vl[-1]

    init_Vhs = jnp.zeros((T + 1, n_agent, nh)).at[0, :].set(Vh_final)
    init_Vl = jnp.zeros(T + 1).at[0].set(Vl_final)[:, None].repeat(n_agent, axis=1)
    init_carry = (init_Vhs, init_Vl, init_gae_coeffs)

    ts = jnp.arange(T)[::-1]
    inps = (ts, Tah_hs, T_l, Tah_Vh, T_Vl)

    _, Qs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    Qhs_GAEs, Ql_GAEs = Qs_GAEs[:, :, :nh], Qs_GAEs[:, 0, nh]
    return assert_shape(Qhs_GAEs, (T, n_agent, nh)), assert_shape(Ql_GAEs, T)
