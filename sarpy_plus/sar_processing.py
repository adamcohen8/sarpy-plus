from sarpy_plus import RadarParams, c, nextpow2, integer_sequence
import numpy as np
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft, fftshift, fftfreq, ifftshift, fft2, ifft2


def crop_ph(ph: np.ndarray,
            radar: RadarParams,
            thresh_frac: float = 0.01):
    """
    Crop phase‑history data to only those pulses containing significant target energy.
    Returns a new RadarParams object with updated t_slow and num_pulses.

    Parameters
    ----------
    ph : np.ndarray, shape (Nr, Na)
        Fast‑time/slow‑time phase‑history matrix.
    radar : RadarParams
        Radar parameters object.
    thresh_frac : float, optional
        Fraction of peak energy used as threshold (default is 0.01).

    Returns
    -------
    ph_cropped : np.ndarray, shape (Nr, n_active)
        Phase‑history cropped to active pulses.
    updated_radar : RadarParams
        New RadarParams object with updated t_slow and num_pulses.
    pulse_inds : np.ndarray, shape (n_active,)
        Indices of the selected active pulses.
    """
    Nr, Na = ph.shape
    prf_hz = radar.prf_hz
    t_slow = radar.t_slow

    # 1) Compute per‑pulse energy and threshold mask
    energy = np.sum(np.abs(ph), axis=0)
    threshold = thresh_frac * energy.max()
    pulse_mask = energy > threshold
    pulse_inds = np.where(pulse_mask)[0]

    # 2) Crop phase history
    ph_cropped = ph[:, pulse_mask]

    # 3) Crop t_slow
    t_slow_cropped = t_slow[pulse_mask]
    num_pulses_cropped = t_slow_cropped.shape[0]

    # 4) Create updated RadarParams — manual copy, update t_slow and num_pulses
    updated_radar = RadarParams(
        platform_altitude_m=radar.platform_altitude_m,
        platform_speed_mps=radar.platform_speed_mps,
        center_frequency_hz=radar.center_frequency_hz,
        range_resolution_m=radar.range_resolution_m,
        pulse_width_sec=radar.pulse_width_sec,
        prf_hz=radar.prf_hz,
        cross_range_resolution_m=radar.cross_range_resolution_m,
        ground_range_swath_m=radar.ground_range_swath_m,
        range_grp_m=radar.range_grp_m,
        azimuth_aperture_factor=radar.azimuth_aperture_factor,
        range_oversample=radar.range_oversample,
        noise_power_db=radar.noise_power_db,
        transmit_power_watts=radar.transmit_power_watts,
        transmit_gain_db=radar.transmit_gain_db,
        receive_gain_db=radar.receive_gain_db,
        system_temperature_K=radar.system_temperature_K,
        noise_figure_db=radar.noise_figure_db,
    )

    object.__setattr__(updated_radar, "t_slow", t_slow_cropped)
    object.__setattr__(updated_radar, "num_pulses", num_pulses_cropped)

    return ph_cropped, updated_radar


def crop_ph_manual(ph: jnp.ndarray, radar: RadarParams, Va: float, t_0: float = 0.0):
    """
    Manually crop phase-history data to the aperture length based on target velocity Va.

    Parameters
    ----------
    radar : RadarParams
        Radar parameters object.
    ph : np.ndarray, shape (Nr, Na)
        Fast-time/slow-time phase-history matrix.
    Va : float
        Target azimuth (along-track) velocity (m/s).
    t_0 : float, optional
        Center time where the target is in the data (default is 0.0).

    Returns
    -------
    ph_cropped : np.ndarray, shape (Nr, n_active)
        Phase-history cropped to active pulses.
    updated_radar : RadarParams
        New RadarParams object with updated t_slow and num_pulses.
    pulse_inds : np.ndarray, shape (n_active,)
        Indices of the selected active pulses.
    """
    Nr, Na = ph.shape
    V_platform = radar.platform_speed_mps
    V_rel = abs(V_platform - Va)  # Relative velocity

    if V_rel == 0:
        raise ValueError("Relative velocity is zero; aperture time undefined.")

    lam = radar.center_wavelength_m
    res_az = radar.cross_range_resolution_m
    R = radar.range_grp_m
    prf = radar.prf_hz

    theta_sar = lam / (2.0 * res_az)
    L = theta_sar * R
    T = L / V_rel  # Required aperture time for desired resolution

    num_pulses_needed = int(np.ceil(T * prf))
    num_pulses_needed = min(num_pulses_needed, Na)  # Don't exceed available

    # Find center index
    t_slow = radar.t_slow
    center_idx = np.argmin(np.abs(t_slow - t_0))

    # Calculate half
    half = num_pulses_needed // 2

    # Determine start and end indices, handling boundaries
    start = max(0, center_idx - half)
    end = min(Na, center_idx + half + (num_pulses_needed % 2))

    # Adjust if the window is too small
    actual_num = end - start
    if actual_num < num_pulses_needed:
        if start == 0:
            end = min(Na, start + num_pulses_needed)
        elif end == Na:
            start = max(0, end - num_pulses_needed)

    pulse_inds = np.arange(start, end)

    # Crop phase history
    ph_cropped = ph[:, pulse_inds]

    # Crop t_slow
    t_slow_cropped = t_slow[pulse_inds]
    num_pulses_cropped = len(pulse_inds)

    # Create updated RadarParams — manual copy, update t_slow and num_pulses
    updated_radar = RadarParams(
        platform_altitude_m=radar.platform_altitude_m,
        platform_speed_mps=radar.platform_speed_mps,
        center_frequency_hz=radar.center_frequency_hz,
        range_resolution_m=radar.range_resolution_m,
        pulse_width_sec=radar.pulse_width_sec,
        prf_hz=radar.prf_hz,
        cross_range_resolution_m=radar.cross_range_resolution_m,
        ground_range_swath_m=radar.ground_range_swath_m,
        range_grp_m=radar.range_grp_m,
        azimuth_aperture_factor=radar.azimuth_aperture_factor,
        range_oversample=radar.range_oversample,
        noise_power_db=radar.noise_power_db,
        transmit_power_watts=radar.transmit_power_watts,
        transmit_gain_db=radar.transmit_gain_db,
        receive_gain_db=radar.receive_gain_db,
        system_temperature_K=radar.system_temperature_K,
        noise_figure_db=radar.noise_figure_db,
        SNR_single=radar.SNR_single,
        noise=radar.noise,
        antenna_pattern=radar.antenna_pattern
    )

    object.__setattr__(updated_radar, "t_slow", t_slow_cropped)
    object.__setattr__(updated_radar, "num_pulses", num_pulses_cropped)

    return ph_cropped, updated_radar



def pulse_compression(ph: jnp.ndarray,
                      radar: RadarParams,
                      mode: str = "same") -> jnp.ndarray:
    """
    Perform range (pulse) compression via an LFM matched filter using JAX FFT.

    Parameters
    ----------
    ph : jnp.ndarray, shape (n_ranges, n_pulses)
        Cropped phase-history data.
    radar : RadarParams
        Radar parameters object.
    mode : str, optional
        Convolution mode ("same" or "full").

    Returns
    -------
    data_pc : jnp.ndarray
        Range-compressed data of same shape as `ph`.
    """
    # Unpack Radar
    fs = radar.sample_rate_hz
    pulse_width_sec = radar.pulse_width_sec
    chirp_rate_hzpsec = radar.chirp_rate_hz_per_sec

    # Number of taps for the matched filter
    Nmf = int(round(pulse_width_sec * fs))

    # Create time axis centered at zero
    t_mf = jnp.linspace(-0.5, 0.5, Nmf) * pulse_width_sec

    # LFM chirp matched filter
    h_rc = jnp.exp(-1j * jnp.pi * chirp_rate_hzpsec * t_mf**2)

    # Prepare for FFT-based convolution
    Nr, Na = ph.shape
    Nfft = Nr + Nmf - 1  # size of FFT

    # Pad inputs to Nfft
    ph_padded = jnp.pad(ph, ((0, Nfft - Nr), (0, 0)))
    h_rc_padded = jnp.pad(h_rc, (0, Nfft - Nmf))

    # FFTs
    PH_f = fft(ph_padded, axis=0)
    H_f = fft(h_rc_padded)[:, None]

    # Multiply in freq domain
    Y_f = PH_f * H_f

    # IFFT to time domain
    y = ifft(Y_f, axis=0)

    # Select output according to mode
    if mode == "same":
        start = (Nmf - 1) // 2
        end = start + Nr
        data_pc = y[start:end, :]
    elif mode == "full":
        data_pc = y
    else:
        raise ValueError("mode must be 'same' or 'full'")

    return data_pc




def remove_dc(data_ks: jnp.ndarray,
              radar: RadarParams) -> (jnp.ndarray, float):
    """
    Estimate and remove the Doppler centroid from slow‑time data using
    the magnitude‑based energy‑balance method. (JAX version)

    Parameters
    ----------
    data_ks : jnp.ndarray, shape (n_ranges, n_pulses)
        Keystoned (range‑walk corrected) phase history.
    radar : RadarParams
        Radar parameters object.

    Returns
    -------
    data_dc : jnp.ndarray, shape like data_ks
        Doppler‑centroid–compensated data.
    fdc_hat : float
        Estimated baseband Doppler centroid (Hz).
    """
    # Unpack radar
    prf_hz = radar.prf_hz

    # number of pulses and PRI
    n_ranges, Na = data_ks.shape
    PRI = 1.0 / prf_hz

    # slow‑time axis centered at zero
    tm = (jnp.arange(Na) - (Na-1)/2) * PRI

    # 1) Azimuthal FFT and average power
    Y_az = fft(data_ks, axis=1)
    P_d  = jnp.mean(jnp.abs(Y_az)**2, axis=0)

    # 2) Power‑balance filter +1 on [0:Na/2), –1 on [Na/2:Na)
    Fpb = jnp.ones(Na)
    Fpb = Fpb.at[Na//2:].set(-1)

    # 3) Circular convolution via FFT → balance output
    balance = ifft(fft(P_d) * fft(Fpb)).real

    # 4) Doppler‑frequency axis (–PRF/2 … +PRF/2)
    fd = fftshift(fftfreq(Na, d=PRI))

    # 5) Find zero‑crossing with negative slope
    zeros = jnp.where((balance[:-1] > 0) & (balance[1:] < 0))[0]

    # Safe index handling (JAX wants static shape control)
    fdc_hat = jnp.where(zeros.size > 0,
                        fd[zeros[0]],
                        fd[jnp.argmax(balance)])

    # 6) Compensate phase in time domain
    phase_dc = jnp.exp(-1j * 2 * jnp.pi * fdc_hat * tm)
    data_dc  = data_ks * phase_dc[jnp.newaxis, :]

    return data_dc, float(fdc_hat)

def remove_dc_HC(data_ks: jnp.ndarray,
              radar: RadarParams,
              Vr: float) -> (jnp.ndarray, float):
    """
    Estimate and remove the Doppler centroid from slow‑time data using
    the magnitude‑based energy‑balance method. (JAX version)

    Parameters
    ----------
    data_ks : jnp.ndarray, shape (n_ranges, n_pulses)
        Keystoned (range‑walk corrected) phase history.
    radar : RadarParams
        Radar parameters object.

    Returns
    -------
    data_dc : jnp.ndarray, shape like data_ks
        Doppler‑centroid–compensated data.
    fdc_hat : float
        Estimated baseband Doppler centroid (Hz).
    """
    # Unpack radar
    prf_hz = radar.prf_hz

    # number of pulses and PRI
    n_ranges, Na = data_ks.shape
    PRI = 1.0 / prf_hz

    # slow‑time axis centered at zero
    tm = (jnp.arange(Na) - (Na-1)/2) * PRI

    # Safe index handling (JAX wants static shape control)
    fdc_hat = -2.0*Vr/radar.center_wavelength_m

    # 6) Compensate phase in time domain
    phase_dc = jnp.exp(-1j * 2 * jnp.pi * fdc_hat * tm)
    data_dc  = data_ks * phase_dc[jnp.newaxis, :]

    return data_dc, float(fdc_hat)



def range_curve_correction_filter(data_dc: jnp.ndarray,
                                  radar: RadarParams) -> jnp.ndarray:
    """
    Apply the range-curve correction filter H_RCM_curve(fr, ta) in the
    (range-frequency, azimuth-time) domain, as per Equation 13.

    Parameters
    ----------
    data_dc : jnp.ndarray, shape (n_ranges, n_pulses)
        Doppler-centroid-corrected time-domain data.
    radar : RadarParams
        Radar parameters object.

    Returns
    -------
    data_rc : jnp.ndarray, shape (n_ranges, n_pulses)
        Range-curve-corrected time-domain data.
    """
    # Unpack radar
    sample_rate_hz = radar.sample_rate_hz
    prf_hz = radar.prf_hz
    platform_speed_mps = radar.platform_speed_mps
    range_grp_m = radar.range_grp_m  # R0

    # Constants and dimensions
    nr, na = data_dc.shape

    # 1) FFT into fr-ta domain (only along range), then shift
    S = fftshift(fft(data_dc, axis=0), axes=0)  # shape (nr, na)

    # 2) Build baseband range frequency axis (fr in Hz)
    fr = fftshift(fftfreq(nr, d=1.0 / sample_rate_hz))

    # 3) Build azimuth time axis (ta in seconds), centered at 0
    ta = jnp.arange(-na // 2, na // 2) / prf_hz  # Assumes even na, pulses centered

    # 4) Construct H_RCM_curve(fr, ta) per Equation 13
    H = jnp.exp(
        1j * 2 * jnp.pi * (platform_speed_mps ** 2 / (c * range_grp_m))
        * fr[:, None] * (ta[None, :] ** 2)
    )

    # 5) Apply filter and invert (IFFT along range)
    S_corr = S * H
    data_rc = ifft(ifftshift(S_corr, axes=0), axis=0)

    return data_rc




def range_curve_correction_filter_full(data_dc: jnp.ndarray,
                                       radar: RadarParams,
                                       Vy_mps: float,
                                       Ar_mps2: float) -> jnp.ndarray:
    """
    Apply the full range-curve correction filter derived from Eq. (12)
    in the (range-frequency, azimuth-time) domain, as per the paper's approach.

    This uses the quadratic term including Vy and Ar (slant-range acceleration),
    applied in fr-ta domain for direct compensation of the ta^2-dependent migration.

    Parameters
    ----------
    data_dc : jnp.ndarray, shape (n_ranges, n_pulses)
        Doppler-centroid–corrected time-domain data.
    radar : RadarParams
        Radar parameters object.
    Vy_mps : float
        Along-track velocity of the target (m/s).
    Ar_mps2 : float
        Cross-track (slant-range) acceleration of the target (m/s²).

    Returns
    -------
    data_rc : jnp.ndarray, shape (n_ranges, n_pulses)
        Range-curve–corrected time-domain data.
    """
    # Unpack radar parameters
    fs = radar.sample_rate_hz
    prf = radar.prf_hz
    Va = radar.platform_speed_mps
    R0 = radar.range_grp_m

    nr, na = data_dc.shape

    # 1) FFT into fr-ta domain (only along range), then shift
    S = fftshift(fft(data_dc, axis=0), axes=0)  # shape (nr, na)

    # 2) Build baseband range frequency axis (fr in Hz)
    fr = fftshift(fftfreq(nr, d=1.0 / fs))

    # 3) Build azimuth time axis (ta in seconds), centered at 0
    ta = jnp.arange(-na // 2, na // 2) / prf  # Assumes even na, pulses centered

    # 4) Compute denominator from Eq. (12)
    denom = (Va - Vy_mps)**2 - R0 * Ar_mps2
    if denom == 0.0:
        raise ValueError("Denominator (Va-Vy)² − R0·Ar is zero; filter undefined.")

    # 5) Construct full H_RCM_curve(fr, ta) generalized from Eq. (13)
    H_full = jnp.exp(
        1j * 2 * jnp.pi * (denom / (c * R0))
        * fr[:, None] * (ta[None, :] ** 2)
    )

    # 6) Apply filter and invert (IFFT along range)
    S_corr = S * H_full
    data_rc = ifft(ifftshift(S_corr, axes=0), axis=0)

    return data_rc




def range_walk_filter(data_rc: jnp.ndarray,
                      Vr: float,
                      radar: RadarParams) -> jnp.ndarray:
    """
    Remove range-walk from range-curve-corrected phase history. (JAX version)

    Parameters:
        data_rc : jnp.ndarray, shape (N_rg, N_az)
            Range-curve-corrected complex phase history.
        Vr      : float
            Estimated cross-track velocity [m/s].
        radar   : RadarParams
            Radar parameters object (uses sample_rate_hz and t_slow).

    Returns:
        data_rw : jnp.ndarray, shape (N_rg, N_az)
            Complex phase history with range-walk removed.
    """
    # 1) FFT in range (fast time) domain
    S_rg = fft(data_rc, axis=0)

    # 2) Build range-frequency axis
    N_rg, N_az = data_rc.shape
    fr = fftfreq(N_rg, d=1.0 / radar.sample_rate_hz)

    # 3) Use radar.t_slow as slow-time axis
    ta = radar.t_slow  # shape (N_az,)

    # 4) Build (f_r, t_a) filter
    # H_tr: shape (N_rg, N_az)
    H_tr = jnp.exp(1j * (4 * jnp.pi * Vr / c) * jnp.outer(fr, ta))

    # 5) Apply filter and IFFT back to time domain
    data_rw = ifft(S_rg * H_tr, axis=0)

    return data_rw





def azimuth_matched_filter(data_rc: jnp.ndarray,
                           Ka_hat: float,
                           radar: RadarParams) -> (jnp.ndarray, jnp.ndarray):
    """
    Apply an azimuthal matched filter (quadratic phase correction) in the time domain,
    then form the final azimuth‐compressed image via slow‑time IFFT. (JAX version)

    Parameters
    ----------
    data_rm : jnp.ndarray, shape (n_ranges, n_pulses)
        Range‑curvature–corrected data in the time domain.
    Ka_hat : float
        Estimated Doppler chirp rate (Hz/s^2).
    radar : RadarParams
        Radar parameters object.

    Returns
    -------
    data_ac : jnp.ndarray, shape (n_ranges, n_pulses)
        Azimuth‑compressed time‑domain data (after phase correction).
    img_final : jnp.ndarray, shape (n_ranges, n_pulses)
        Final focused image obtained by IFFT along azimuth.
    """
    # Unpack radar
    _, num_pulses = data_rc.shape
    num_fft = num_pulses

    # 1) FFT along azimuth, centered
    data_rc_f = fftshift(fft(ifftshift(data_rc, axes=1), num_fft, axis=1), axes=1)

    # 2) Build Doppler frequency axis
    prf_hz = radar.prf_hz
    doppler_axis_hz = jnp.linspace(-num_pulses / 2, num_pulses / 2, num_pulses) * (prf_hz / num_pulses)

    # 3) Quadratic phase correction (matched filter)
    phase_ac = jnp.exp(-1j * (jnp.pi / Ka_hat) * doppler_axis_hz**2.0)
    data_ac_f = data_rc_f * phase_ac[jnp.newaxis, :]

    # 4) Inverse FFT along azimuth (slow time), centered
    img_final = fftshift(ifft(ifftshift(data_ac_f, axes=1), num_fft, axis=1), axes=1)

    # Exit function
    return data_ac_f, img_final


def azimuth_matched_filter_cubic(data_rc: jnp.ndarray,
                                 radar: RadarParams,
                                 Vr: float,
                                 Va: float,
                                 Ar: float,
                                 Aa: float) -> (jnp.ndarray, jnp.ndarray):
    """
    Apply an azimuthal matched filter (quadratic and cubic phase correction) in the Doppler domain,
    then form the final azimuth-compressed image via slow-time IFFT. (JAX version)

    Parameters
    ----------
    data_rc : jnp.ndarray, shape (n_ranges, n_pulses)
        Range-curvature-corrected data in the time domain.
    Vr : float
        Radial (slant-range) velocity of the target (m/s).
    Vy : float
        Along-track velocity of the target (m/s).
    Ar : float
        Slant-range acceleration of the target (m/s²).
    radar : RadarParams
        Radar parameters object.

    Returns
    -------
    data_ac_f : jnp.ndarray, shape (n_ranges, n_pulses)
        Data after phase correction in the Doppler domain.
    img_final : jnp.ndarray, shape (n_ranges, n_pulses)
        Final focused image obtained by IFFT along azimuth.
    """
    # Unpack radar
    lam = radar.center_wavelength_m
    Vp = radar.platform_speed_mps
    R0 = radar.range_grp_m
    prf_hz = radar.prf_hz
    _, num_pulses = data_rc.shape
    num_fft = num_pulses
    ta = radar.t_slow

    # Compute second- and third-order Doppler parameters (k2 and k3)
    k2 = ((Vp-Va)**2 - Ar*R0) / (2.0 * R0)
    k3 = -(Vr * (Vp-Va)**2) / (2.0 * R0**2) + (Aa*(Vp-Va)) / (2.0 * R0)

    # 1) FFT along azimuth, centered
    data_rc_f = fftshift(fft(ifftshift(data_rc, axes=1), axis=1), axes=1)

    # 2) Quadratic and cubic phase correction (matched filter from equation 21)

    h = jnp.exp(1j * (4*jnp.pi/radar.center_wavelength_m) * (k2*ta**2 + k3*ta**3))
    H = fftshift(fft(ifftshift(h))) #/ jnp.sqrt(float(num_pulses))
    H = H / (jnp.abs(H) + 1e-12)
    data_ac_f = data_rc_f * H[jnp.newaxis, :]

    # 3) Inverse FFT along azimuth (slow time), centered
    img_final = fftshift(ifft(ifftshift(data_ac_f,axes=1), axis=1), axes=1)

    # Exit function
    return data_ac_f, img_final





def sinc_interp(inp, x_in, x_new, N, win):
    """
    Sinc-based (band-limited) interpolation.
    inp   : 1D array of input samples
    x_in  : 1D array of sample positions (uniformly spaced)
    x_new : 1D array of desired sample positions (uniform spacing)
    N     : filter order (odd integer), in units of max(dx_in,dx_out)
    win   : 1 to enable Hamming window on kernel, 0 to skip windowing

    returns (out, x_out) where
      x_out is a subset of x_new that could be safely interpolated,
      out   are the corresponding interpolated values.
    """
    #--- sanity checks -----------------------------
    if N % 2 == 0:
        raise ValueError(f"sinc_interp: filter order N={N} must be odd")
    if inp.ndim != 1 or x_in.ndim != 1:
        raise ValueError("sinc_interp: inp and x_in must be 1D")
    if x_in.size != inp.size:
        raise ValueError("sinc_interp: x_in and inp must be same length")
    # uniform spacing?
    dx_in  = x_in[1] - x_in[0]
    dx_out = x_new[1] - x_new[0]
    # half‐length of kernel in samples
    Nhalf = (N - 1) // 2
    # choose larger spacing to set bandwidth
    delta = max(dx_in, dx_out)

    # figure which x_new can be interpolated without running off the ends
    valid = np.where(
        (x_new - Nhalf*delta >= x_in[0]) &
        (x_new + Nhalf*delta <= x_in[-1])
    )[0]
    if valid.size == 0:
        return np.array([]), np.array([])

    x_out = x_new[valid]
    out   = np.zeros(x_out.shape, dtype=inp.dtype)

    # main loop: one output sample at a time
    eps = np.finfo(float).eps
    for idx, xc in enumerate(x_out):
        # determine which input samples feed this kernel:
        lo = xc - Nhalf*delta
        hi = xc + Nhalf*delta
        rel_ix = np.nonzero((x_in >= lo) & (x_in <= hi))[0]
        x_rel   = x_in[rel_ix] - xc
        # avoid zero‐division in sinc
        zero_locs = np.isclose(x_rel, 0.0)
        if zero_locs.any():
            x_rel[zero_locs] = eps

        # ideal sinc kernel in continuous time:
        #   h(t) = sin(pi*t/del)/(pi*t/del)
        h = np.sinc(x_rel / delta)

        # optional Hamming window of same length:
        if win:
            # continuous‐time Hamming: w(t)=0.54+0.46*cos(pi*t/del)
            # Richards divides by (Nhalf+1) to scale the cos argument:
            w = 0.54 + 0.46 * np.cos(np.pi * (x_rel/delta) / (Nhalf+1))
            h *= w

        # normalize kernel area:
        h /= h.sum()

        # accumulate:
        out[idx] = np.dot(inp[rel_ix], h)

    return out, x_out

def keystone_transform_richards(cphd_time_matrix,
                                radar: RadarParams,
                                sinc_order: int = 11,
                                doppler_ambiguity=0):
    '''
    This function implements the Keystone Transform that is used for range walk correction. Range walk occurs when the target traverses over
    multiple range bins during the coherent processing interval.
    '''

    #Unpack Radar
    center_frequency_hz = radar.center_frequency_hz
    sample_rate_hz = radar.sample_rate_hz

    # complex-valued phase history matrix
    dims_cphd = np.shape(cphd_time_matrix)
    if len(dims_cphd) > 2:
        raise ValueError("<error: keystone_transform> input must be a phase history matrix, not a radar data cube...")

    # number of range bins and pulses
    num_ranges = dims_cphd[0]
    num_range_fft = int(nextpow2(num_ranges))
    num_pulses = dims_cphd[1]
    pulse_index = integer_sequence(num_pulses, True)

    # convert to fast-time frequency/slow-time matrix
    rngfreq_slowtime_matrix = np.fft.fftshift(np.fft.fft(cphd_time_matrix, num_range_fft, 0), 0)
    #rngfreq_baseband_hz = column_vector(integer_sequence(num_range_fft, True)) * (
                #sample_rate_hz / num_range_fft)

    # build true baseband frequencies, then shift them to match fftshifted data
    freqs = np.fft.fftfreq(num_range_fft, d=1 / sample_rate_hz)  # array of [0, +, –, …]
    rngfreq_baseband_hz = np.fft.fftshift(freqs)[:, None]  # now it runs –Fs/2→+Fs/2

    # keystone transformation
    keystone_matrix = np.zeros((num_range_fft, num_pulses), dtype='complex')
    for ii in range(0, num_range_fft):

        scale_val = center_frequency_hz / (center_frequency_hz + rngfreq_baseband_hz[ii])

        y_tmp, x_tmp = sinc_interp(rngfreq_slowtime_matrix[ii, :], pulse_index, scale_val * pulse_index,
                                                sinc_order, 1)

        mi = np.size(y_tmp)
        dm = num_pulses - mi
        new_index = np.arange(int(dm / 2), int(dm / 2 + mi))

        # after sinc_interp:
        order = np.argsort(x_tmp)
        y_sorted = y_tmp[order]
        # ensure new_index is ascending
        new_index = np.sort(new_index)
        keystone_matrix[ii, new_index] = y_sorted

        # keystone transformed range slice
        #new_index = np.arange(int(1 + dm / 2 - 1), int(1 + dm / 2 + mi - 1), 1)
        #keystone_matrix[ii, new_index] = y_tmp

    # adjust for doppler ambiguity number
    if doppler_ambiguity > 0:
        for ii in range(0, num_pulses):
            for jj in range(0, num_range_fft):
                scale_val = center_frequency_hz / (center_frequency_hz + rngfreq_baseband_hz[ii])
                keystone_matrix[jj, ii] = keystone_matrix[jj, ii] * np.exp(
                    1j * 2.0 * doppler_ambiguity * pulse_index[ii] * scale_val)

    # convert back to fast-time/slow-time matrix and retain the first num_ranges bins along the range dimension
    cphd_corrected_matrix = np.fft.ifft(np.fft.ifftshift(keystone_matrix, 0), num_range_fft, 0)
    cphd_corrected_matrix = cphd_corrected_matrix[0:num_ranges, :]

    # exit function
    return cphd_corrected_matrix



def range_curve_correction_filter_inv(data_dc: jnp.ndarray,
                                  radar: RadarParams) -> jnp.ndarray:
    """
    Apply the range-curve correction filter H_RCM_curve(fr, ta) in the
    (range-frequency, azimuth-time) domain, as per Equation 13.

    Parameters
    ----------
    data_dc : jnp.ndarray, shape (n_ranges, n_pulses)
        Doppler-centroid-corrected time-domain data.
    radar : RadarParams
        Radar parameters object.

    Returns
    -------
    data_rc : jnp.ndarray, shape (n_ranges, n_pulses)
        Range-curve-corrected time-domain data.
    """
    # Unpack radar
    sample_rate_hz = radar.sample_rate_hz
    prf_hz = radar.prf_hz
    platform_speed_mps = radar.platform_speed_mps
    range_grp_m = radar.range_grp_m  # R0

    # Constants and dimensions
    nr, na = data_dc.shape

    # 1) FFT into fr-ta domain (only along range), then shift
    S = fftshift(fft(data_dc, axis=0), axes=0)  # shape (nr, na)

    # 2) Build baseband range frequency axis (fr in Hz)
    fr = fftshift(fftfreq(nr, d=1.0 / sample_rate_hz))

    # 3) Build azimuth time axis (ta in seconds), centered at 0
    ta = jnp.arange(-na // 2, na // 2) / prf_hz  # Assumes even na, pulses centered

    # 4) Construct H_RCM_curve(fr, ta) per Equation 13
    H = jnp.exp(
        -1j * 2 * jnp.pi * (platform_speed_mps ** 2 / (c * range_grp_m))
        * fr[:, None] * (ta[None, :] ** 2)
    )

    # 5) Apply filter and invert (IFFT along range)
    S_corr = S * H
    data_rc = ifft(ifftshift(S_corr, axes=0), axis=0)

    return data_rc


