import jax.numpy as jnp
import jax
from sarpy_plus import RadarParams, TargetParams, c, k




def SAR_Sim(radar: RadarParams, tgt: TargetParams,
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
