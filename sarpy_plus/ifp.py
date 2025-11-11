from sarpy_plus.params import RadarParams
from sarpy_plus.sar_processing import (pulse_compression,
                                       remove_dc,
                                       range_curve_correction_filter,
                                       azimuth_matched_filter)
from sarpy_plus.constants import c
import jax.numpy as jnp
from jax import vmap

# Range Doppler Algorithm

def rda(ph: jnp.ndarray, radar: RadarParams) -> jnp.ndarray:
    """
    Range-Doppler Algorithm (RDA) for stripmap SAR image formation.

    Parameters
    ----------
    ph : jnp.ndarray, shape (n_ranges, n_pulses)
        Raw phase history data.
    radar : RadarParams
        Radar parameters object.

    Returns
    -------
    image : jnp.ndarray
        Final focused SAR image.
    """
    # 1) Range compression
    data_pc = pulse_compression(ph, radar)

    # 2) Doppler centroid removal
    # data_dc, dc_hat = remove_dc(data_pc, radar)

    # 3) Range-curve correction
    data_rc = range_curve_correction_filter(data_pc, radar)

    # 4) Azimuth matched filtering
    Ka = (2.0 * radar.platform_speed_mps**2) / (
        radar.center_wavelength_m * radar.range_grp_m
    )
    data_ac_f, image = azimuth_matched_filter(data_rc, Ka, radar)

    return image

# Range Migration Algorithm / Omega K Algorithm

def wka(ph: jnp.ndarray, radar: RadarParams) -> jnp.ndarray:
    """
    Focuses radar quadrature-demodulated phase history using the Omega-K Algorithm with JAX.

    Parameters:
    ph (jax.numpy.ndarray): Raw phase history data [num_ranges x num_pulses]
    radar (dict): Radar/platform parameters containing:
        - 'sample_rate_hz': ADC sample rate
        - 'center_frequency_hz': Carrier center frequency (Hz)
        - 'prf_hz': Pulse repetition frequency (Hz)
        - 'pulse_width_sec': Waveform pulse width (s)
        - 'bandwidth_hz': Waveform bandwidth (Hz)
        - 'platform_speed_mps': SAR platform velocity (m/s)
        - 'range_grp_m': Slant range to the ground reference point (m)

    Returns:
    slc_image_wka (jax.numpy.ndarray): Focused Single Look Complex (SLC) image (complex 2D array)
    data_sm (jax.numpy.ndarray): Stolt-mapped data (for debugging or further processing)

    Notes:
    - This version uses linear interpolation for Stolt mapping to ensure JAX compatibility.
    - Omits any dB or magnitude conversions.
    - Ensure 'radar' contains all the required keys.

    Example:
        slc_image, data_sm = wka(ph, radar)
    """

    # --- 2) Basic Size & Frequency Vectors ---
    num_ranges, num_pulses = ph.shape

    range_freq = jnp.linspace(-0.5, 0.5, num_ranges) * radar.sample_rate_hz
    az_freq = jnp.linspace(-0.5, 0.5, num_pulses) * radar.prf_hz

    # --- 3) 2D FFT of Phase History ---
    data_fft2 = jnp.fft.fftshift(jnp.fft.fft2(ph))

    # --- 4) Bulk Compression (Range Frequency Migration) ---
    az_freq_mat, range_freq_mat = jnp.meshgrid(az_freq, range_freq)

    chirp_rate_hzpsec = radar.bandwidth_hz / radar.pulse_width_sec

    phase_RFM = (4 * jnp.pi * radar.range_grp_m / c) * \
                jnp.sqrt((radar.center_frequency_hz + range_freq_mat) ** 2 -
                         (c ** 2 * az_freq_mat ** 2) / (4 * radar.platform_speed_mps ** 2)) + \
                (jnp.pi * range_freq_mat ** 2 / chirp_rate_hzpsec)

    data_BC = data_fft2 * jnp.exp(1j * phase_RFM)

    # --- 5) Stolt Mapping (Range Cell Migration Correction in Frequency) ---
    def interp_column(az_f, col):
        F_interp = jnp.sqrt((radar.center_frequency_hz + range_freq) ** 2 -
                            (c ** 2 * az_f ** 2) / (4 * radar.platform_speed_mps ** 2)) - \
                   radar.center_frequency_hz
        return jnp.interp(F_interp, range_freq, col)

    vmapped_interp = vmap(interp_column, in_axes=(0, 0))
    data_SM = vmapped_interp(az_freq, data_BC.T).T

    data_SM = jnp.nan_to_num(data_SM, nan=jnp.finfo(jnp.float64).eps)

    # --- Extra Phase Shift to re-center image at range_grp_m ---
    k_range_mat = (4 * jnp.pi / c) * \
                  (radar.center_frequency_hz + range_freq_mat)

    phase_shift_mat = jnp.exp(-1j * k_range_mat * radar.range_grp_m)

    data_SM = data_SM * phase_shift_mat

    # --- 6) 2D IFFT to form the final image ---
    slc_image_wka = jnp.fft.ifft2(jnp.fft.ifftshift(data_SM))

    return slc_image_wka




# Chirp Scaling Algorithm



# Polar Format Algorithm



# Back Projection Algorithm