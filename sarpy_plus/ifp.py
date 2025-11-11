from sarpy_plus.params import RadarParams
from sarpy_plus.sar_processing import (pulse_compression,
                                       remove_dc,
                                       range_curve_correction_filter,
                                       azimuth_matched_filter)
import jax.numpy as jnp

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

    return 0


# Chirp Scaling Algorithm



# Polar Format Algorithm



# Back Projection Algorithm