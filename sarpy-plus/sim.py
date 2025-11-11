import jax.numpy as jnp
import jax
from SARpy.constants import c, k
from SARpy.params import RadarParams, TargetParams


def SAR_Sim_Extended(radar: RadarParams,
                     tgt: TargetParams) -> jnp.ndarray:
    """
    Vectorized SAR raw phase history simulator with JAX.

    Parameters
    ----------
    radar : RadarParams
        Radar/trajectory parameters.
    tgt : TargetParams
        Target positions, velocities, etc.
    range_oversample : int
        Oversampling factor for the fast‑time sampling rate.

    Returns
    -------
    phase_history : jnp.ndarray [num_ranges, num_pulses] (complex64)
    """
    fs = radar.sample_rate_hz
    lam = radar.center_wavelength_m
    sar_angle_rad = radar.beamwidth_rad
    t_slow = radar.t_slow
    t_fast = radar.t_fast
    num_pulses = radar.num_pulses

    grazing = jnp.arcsin(radar.platform_altitude_m / radar.range_grp_m)

    mid_pos = jnp.array([-radar.range_grp_m * jnp.cos(grazing), 0.0, radar.platform_altitude_m])
    sar_pos = jnp.stack((
        jnp.full(num_pulses, mid_pos[0]),
        t_slow * radar.platform_speed_mps,
        jnp.full(num_pulses, mid_pos[2])
    ), axis=0)  # (3, num_pulses)

    N_targets = tgt.rcs_dbsm.size
    phase0 = tgt.phase_rad if tgt.phase_rad is not None else jnp.zeros_like(tgt.rcs_dbsm)

    t_slow_broadcast = t_slow[None, :]
    r0 = (tgt.positions_m[:, :, None] +
          tgt.velocities_mps[:, :, None] * t_slow_broadcast +
          0.5 * tgt.accelerations_mps2[:, :, None] * t_slow_broadcast**2)

    rel = r0 - sar_pos[:, None, :]  # (3, N_targets, N_pulses)
    R = jnp.linalg.norm(rel, axis=0)  # (N_targets, N_pulses)

    beam_dir = jnp.array([jnp.cos(grazing), 0.0, -jnp.sin(grazing)])  # unit vector
    cosang = jnp.einsum("ijk,i->jk", rel, beam_dir) / (R * jnp.linalg.norm(beam_dir))
    theta = jnp.arccos(cosang)
    beam_mask = theta <= (sar_angle_rad / 2.0)  # (N_targets, N_pulses)

    tau = 2.0 * R / c  # (N_targets, N_pulses)
    t_cen = t_fast[:, None, None] - tau[None, :, :]  # (num_ranges, N_targets, N_pulses)
    mask = jnp.abs(t_cen) < (radar.pulse_width_sec / 2.0)  # (num_ranges, N_targets, N_pulses)

    p1 = -4.0 * jnp.pi * R / lam  # (N_targets, N_pulses)
    p2 = jnp.pi * radar.chirp_rate_hz_per_sec * t_cen**2  # (num_ranges, N_targets, N_pulses)

    total_mask = mask & beam_mask[None, :, :]  # (num_ranges, N_targets, N_pulses)
    phase0_broadcast = phase0[:, None]  # (N_targets, 1)
    total_phase = p1 + phase0_broadcast + p2  # (num_ranges, N_targets, N_pulses)

    target_contrib = jnp.exp(1j * total_phase) * total_mask.astype(jnp.float32)  # (num_ranges, N_targets, N_pulses)
    ph = jnp.sum(target_contrib, axis=1)  # (num_ranges, N_pulses)

    return ph


def SAR_Sim_HF(radar: RadarParams, tgt: TargetParams,
               key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> jnp.ndarray:
    """
    High-fidelity SAR raw phase-history simulator (complex voltage).
    Implements the full monostatic radar-range equation and kTB noise.
    """
    # ---------- Pre-compute constants ----------
    fs      = radar.sample_rate_hz
    lam     = radar.center_wavelength_m
    four_pi = 4.0 * jnp.pi
    lam_fac = lam / (four_pi ** 1.5)       #  λ / (4π)^{3/2}
    t_fast  = radar.t_fast
    t_slow  = radar.t_slow
    Nrg     = t_fast.size
    Np      = t_slow.size

    # ---------- Platform state ----------
    grazing = jnp.arcsin(radar.platform_altitude_m / radar.range_grp_m)
    mid_pos = jnp.array([-radar.range_grp_m * jnp.cos(grazing),
                          0.0,
                          radar.platform_altitude_m])

    sar_pos = jnp.stack(
        (jnp.full(Np, mid_pos[0]),
         t_slow * radar.platform_speed_mps,
         jnp.full(Np, mid_pos[2])), axis=0)        # (3, Np)

    # ---------- Target state ----------
    Ntgt    = tgt.rcs_dbsm.size
    phase0  = (tgt.phase_rad
               if tgt.phase_rad is not None else jnp.zeros_like(tgt.rcs_dbsm))
    t_slow_b = t_slow[None, :]

    r0 = (tgt.positions_m[:, :, None] +
          tgt.velocities_mps[:, :, None] * t_slow_b +
          0.5 * tgt.accelerations_mps2[:, :, None] * t_slow_b ** 2)

    rel = r0 - sar_pos[:, None, :]              # (3, Ntgt, Np)
    R   = jnp.linalg.norm(rel, axis=0)          # (Ntgt, Np)

    # ---------- Antenna pattern ----------
    beam_dir = jnp.array([jnp.cos(grazing), 0., -jnp.sin(grazing)])
    theta    = jnp.arccos(
        jnp.einsum("ijk,i->jk", rel, beam_dir) /
        (R * jnp.linalg.norm(beam_dir)))
    theta_3dB = radar.beamwidth_rad / 2.0
    # G_beam_v  = jnp.sqrt((jnp.sinc(theta / theta_3dB)) ** 2)   # voltage gain

    if radar.antenna_pattern == "binary":
        G_beam_v = (theta <= theta_3dB).astype(jnp.float32)
    elif radar.antenna_pattern == "parabolic":
        u = jnp.pi * theta / theta_3dB
        pattern = jnp.where(u < 1e-6, 1.0, 3 * (jnp.sin(u) - u * jnp.cos(u)) / u ** 3)
        G_beam_v = jnp.abs(pattern)  # Ensure positive, though typically is in main lobe
    elif radar.antenna_pattern == "gaussian":
        G_beam_v = jnp.exp(- (jnp.log(2) / 2) * (theta / theta_3dB) ** 2)
    else:  # Default to "sinc"
        G_beam_v = jnp.sqrt((jnp.sinc(theta / theta_3dB)) ** 2)

    # ---------- Voltage-range equation ----------
    sigma_v = jnp.sqrt(10.0 ** (tgt.rcs_dbsm / 10.0))[:, None]   # (Ntgt,1)
    Gt_lin  = 10.0 ** (radar.transmit_gain_db / 10.0)
    Gr_lin  = 10.0 ** (radar.receive_gain_db / 10.0)

    A0 = (jnp.sqrt(radar.transmit_power_watts * Gt_lin * Gr_lin) *
          lam_fac)                                            # constant part

    A_R = A0 * sigma_v * G_beam_v * R ** (-2)                 # (Ntgt,Np)

    # ---------- Fast-time envelope ----------
    tau  = 2.0 * R / c
    tcen = t_fast[:, None, None] - tau[None, :, :]
    mask = jnp.abs(tcen) < radar.pulse_width_sec / 2.0

    p1 = -4.0 * jnp.pi * R                # 2-way phase (divide by λ later)
    p2 = jnp.pi * radar.chirp_rate_hz_per_sec * tcen ** 2
    total_phase = (p1 / lam) + phase0[:, None] + p2           # (Nr,Ntgt,Np)

    target_v = A_R[None, :, :] * jnp.exp(1j * total_phase) * mask.astype(jnp.float32)
    ph = jnp.sum(target_v, axis=1)                            # (Nr, Np)

    # ---------- kTB noise (complex) ----------
    Tsys  = radar.system_temperature_K
    F_lin = 10.0 ** (radar.noise_figure_db / 10.0)
    N0    = k * Tsys * F_lin              # W / Hz
    Pn    = N0 * fs                         # total noise power in bandwidth
    noise_std = jnp.sqrt(Pn / 2.0)          # per real component

    key, k_r, k_i = jax.random.split(key, 3)
    noise_real = jax.random.normal(k_r, ph.shape) * noise_std
    noise_imag = jax.random.normal(k_i, ph.shape) * noise_std

    if radar.noise:
        ph = ph + (noise_real + 1j * noise_imag)

    return ph






# def SAR_Sim_TS(radar: RadarParams,
#                tgt: TargetParams) -> jnp.ndarray:
#     """
#     SAR raw phase history simulator implementing the approximate signal model
#     from the paper "Imaging and Parameter Estimation of Fast-Moving Targets
#     With Single-Antenna SAR" after range compression.
#
#     Parameters
#     ----------
#     radar : RadarParams
#         Radar/trajectory parameters.
#     tgt : TargetParams
#         Target positions, velocities, etc.
#
#     Returns
#     -------
#     phase_history : jnp.ndarray [num_ranges, num_pulses] (complex)
#         Post-range-compressed phase history.
#     """
#     H = radar.platform_altitude_m
#     Va = radar.platform_speed_mps
#     lam = radar.center_wavelength_m
#     Br = radar.bandwidth_hz
#     Kr = radar.chirp_rate_hz_per_sec
#     t_fast = radar.t_fast
#     t_slow = radar.t_slow
#     num_ranges, num_pulses = t_fast.size, t_slow.size
#
#     grazing = jnp.arcsin(H / radar.range_grp_m)
#     mid_pos = jnp.array([-radar.range_grp_m * jnp.cos(grazing), 0.0, H])
#
#     N_targets = tgt.rcs_dbsm.size
#     phase0 = tgt.phase_rad if tgt.phase_rad is not None else jnp.zeros(N_targets)
#
#     # Compute R0, X, Vr, ar, Vy for each target (assuming z=0)
#     init_pos = tgt.positions_m[:, :]
#     R0 = jnp.linalg.norm(init_pos - mid_pos[:, None], axis=0)  # (N,)
#     X = jnp.sqrt(R0**2 - H**2 + 1e-10)  # Avoid div0, (N,)
#     Vx = tgt.velocities_mps[0, :]
#     ax = tgt.accelerations_mps2[0, :]
#     Vr = Vx * (X / R0)
#     ar = ax * (X / R0)
#     Vy = tgt.velocities_mps[1, :]
#
#     # Time grids
#     tr = t_fast[:, None]  # (num_ranges, 1)
#     ta = t_slow[None, :]  # (1, num_pulses)
#     ta2 = ta ** 2
#     ta3 = ta ** 3
#     tr_ta = tr * ta  # (num_ranges, num_pulses)
#     tr_ta2 = tr * ta2  # (num_ranges, num_pulses)
#
#     # Doppler phases (independent of tr), shape (N_targets, num_pulses)
#     dop_cent = (4 * jnp.pi / lam * Vr[:, None]) * ta
#     dop_rate = (-4 * jnp.pi / lam * ((Va - Vy)**2 - R0 * ar) / (2 * R0))[:, None] * ta2
#     dop_third = (-4 * jnp.pi / lam * Vr * (Va - Vy)**2 / (2 * R0**2))[:, None] * ta3
#     dop_const = (-4 * jnp.pi / lam * R0[:, None]) + phase0[:, None]
#
#     dop_total = dop_cent + dop_rate + dop_third + dop_const  # (N, num_pulses)
#
#     # RCM phases, shape (N_targets, num_ranges, num_pulses)
#     rcm1_tr_ta = 4 * jnp.pi * Kr / c * tr_ta  # (num_ranges, num_pulses)
#     rcm1 = Vr[:, None, None] * rcm1_tr_ta[None, :, :]  # (N, num_ranges, num_pulses)
#
#     coeff_rcm2 = -4 * jnp.pi * Kr / c * ((Va - Vy)**2 - R0 * ar) / (2 * R0)  # (N,)
#     rcm2 = coeff_rcm2[:, None, None] * tr_ta2[None, :, :]  # (N, num_ranges, num_pulses)
#
#     total_phase = dop_total[:, None, :] + rcm1 + rcm2  # (N, num_ranges, num_pulses)
#
#     # Sinc envelope, shape (N_targets, num_ranges, num_pulses)
#     delta_tr = t_fast[:, None] - 2 * R0[None, :] / c  # (num_ranges, N)
#     sinc_part = jnp.sinc(Br * delta_tr)  # (num_ranges, N)
#     sinc_3d = sinc_part.T[:, :, None]  # (N, num_ranges, 1)
#
#     # Contributions and sum
#     contrib = sinc_3d * jnp.exp(-1j * total_phase)  # (N, num_ranges, num_pulses)
#     ph = jnp.sum(contrib, axis=0)  # (num_ranges, num_pulses)
#
#     return -ph

import jax.numpy as jnp


def SAR_Sim_TS(radar: RadarParams,
               tgt: TargetParams) -> jnp.ndarray:
    """
    SAR raw phase history simulator implementing the approximate signal model
    from the paper "Imaging and Parameter Estimation of Fast-Moving Targets
    With Single-Antenna SAR" after range compression.

    Parameters
    ----------
    radar : RadarParams
        Radar/trajectory parameters.
    tgt : TargetParams
        Target positions, velocities, etc.

    Returns
    -------
    phase_history : jnp.ndarray [num_ranges, num_pulses] (complex)
        Post-range-compressed phase history.
    """
    H = radar.platform_altitude_m
    Va = radar.platform_speed_mps
    lam = radar.center_wavelength_m
    Br = radar.bandwidth_hz
    Kr = radar.chirp_rate_hz_per_sec
    t_fast = radar.t_fast
    t_slow = radar.t_slow
    num_ranges, num_pulses = t_fast.size, t_slow.size

    grazing = jnp.arcsin(H / radar.range_grp_m)
    mid_pos = jnp.array([-radar.range_grp_m * jnp.cos(grazing), 0.0, H])

    N_targets = tgt.rcs_dbsm.size
    phase0 = tgt.phase_rad if tgt.phase_rad is not None else jnp.zeros(N_targets)
    A0 = 10 ** (tgt.rcs_dbsm / 10)  # Amplitude from RCS (linear); set to jnp.ones(N_targets) if not needed

    # Compute R0, X, Vr, ar, Vy for each target (assuming z=0)
    init_pos = tgt.positions_m[:, :]
    R0 = jnp.linalg.norm(init_pos - mid_pos[:, None], axis=0)  # (N,)
    X = jnp.sqrt(R0**2 - H**2 + 1e-10)  # Avoid div0, (N,)
    Vx = tgt.velocities_mps[0, :]
    ax = tgt.accelerations_mps2[0, :]
    Vr = Vx * (X / R0)
    ar = ax * (X / R0)
    Vy = tgt.velocities_mps[1, :]

    # Time grids
    tr = t_fast[:, None, None]  # (num_ranges, 1, 1)
    ta = t_slow[None, :]  # (1, num_pulses)
    ta2 = ta ** 2
    ta3 = ta ** 3

    # Approximate R(ta) per equation (4), shape (N_targets, num_pulses)
    coeff2 = ((Va - Vy[:, None]) ** 2 - ar[:, None] * R0[:, None]) / (2 * R0[:, None])
    coeff3 = Vr[:, None] * (Va - Vy[:, None]) ** 2 / (2 * R0[:, None] ** 2)
    R_ta = R0[:, None] - Vr[:, None] * ta + coeff2 * ta2 + coeff3 * ta3

    # Doppler phases (independent of tr), shape (N_targets, num_pulses)
    dop_cent = (4 * jnp.pi / lam * Vr[:, None]) * ta
    dop_rate = (-4 * jnp.pi / lam * ((Va - Vy[:, None]) ** 2 - R0[:, None] * ar[:, None]) / (2 * R0[:, None])) * ta2
    dop_third = (-4 * jnp.pi / lam * Vr[:, None] * (Va - Vy[:, None]) ** 2 / (2 * R0[:, None] ** 2)) * ta3
    dop_const = (-4 * jnp.pi / lam * R0[:, None]) + phase0[:, None]

    dop_total = dop_cent + dop_rate + dop_third + dop_const  # (N, num_pulses)

    # Sinc envelope, now ta-dependent, shape (num_ranges, N_targets, num_pulses)
    delta_tr = tr - (2 / c) * R_ta[None, :, :]  # (num_ranges, N, num_pulses)
    sinc_part = jnp.sinc(Br * delta_tr)  # (num_ranges, N, num_pulses)

    # Contributions and sum
    total_phase = dop_total  # No rcm1/rcm2 needed
    contrib = sinc_part * jnp.exp(1j * total_phase[None, :, :])  # (num_ranges, N, num_pulses)
    ph = jnp.sum(contrib, axis=1)  # Sum over targets (N), (num_ranges, num_pulses)

    return -ph