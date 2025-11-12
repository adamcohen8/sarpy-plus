# params.py
from dataclasses import dataclass, field
from typing import Literal
import jax.numpy as jnp
from sarpy_plus import c, k


@dataclass(frozen=True)
class RadarParams:
    platform_altitude_m: float
    platform_speed_mps: float
    center_frequency_hz: float
    range_resolution_m: float
    pulse_width_sec: float
    prf_hz: float
    cross_range_resolution_m: float
    ground_range_swath_m: float
    range_grp_m: float
    azimuth_aperture_factor: float = 3.0
    range_oversample: float = 2.0
    noise_power_db: float = 0.0
    transmit_power_watts: float = 100.0
    transmit_gain_db: float = 10.0
    receive_gain_db: float = 10.0
    system_temperature_K: float = 290
    noise_figure_db: float = 2.0
    SNR_single: float = None
    SNR_SAR: float = None
    noise: bool = True
    antenna_pattern: str = "sinc"
    demodulation: Literal['Quadrature', 'Dechirp'] = 'Quadrature'


    # Derived
    center_wavelength_m: float = field(init=False)
    bandwidth_hz: float = field(init=False)
    chirp_rate_hz_per_sec: float = field(init=False)
    sample_rate_hz: float = field(init=False)
    num_pulses: int = field(init=False)
    t_fast: jnp.ndarray = field(init=False)
    t_slow: jnp.ndarray = field(init=False)
    beamwidth_rad: float = field(init=False)

    def __post_init__(self):
        lam = c / self.center_frequency_hz
        bw = c / (2.0 * self.range_resolution_m)
        chirp = bw / self.pulse_width_sec
        fs = self.range_oversample * bw

        aperture_time = (
            self.range_grp_m * lam
            / (2.0 * self.platform_speed_mps * self.cross_range_resolution_m)
        )
        total_time = self.azimuth_aperture_factor * aperture_time
        num_pulses = int(jnp.ceil(total_time * self.prf_hz))
        t_slow = (jnp.arange(num_pulses) - (num_pulses - 1) / 2) / self.prf_hz
        azimuth_pixel_m = self.platform_speed_mps / self.prf_hz

        range_pixel_m = c / (2.0 * fs)
        num_ranges = int(round(self.ground_range_swath_m / range_pixel_m))
        t_grp = 2.0 * self.range_grp_m / c
        t_fast = (jnp.arange(num_ranges) - (num_ranges - 1) / 2) / fs + t_grp

        theta_synth = lam / (2.0 * self.cross_range_resolution_m)
        beamwidth_rad = theta_synth

        object.__setattr__(self, "center_wavelength_m", lam)
        object.__setattr__(self, "bandwidth_hz", bw)
        object.__setattr__(self, "chirp_rate_hz_per_sec", chirp)
        object.__setattr__(self, "sample_rate_hz", fs)
        object.__setattr__(self, "num_pulses", num_pulses)
        object.__setattr__(self, "t_slow", t_slow)
        object.__setattr__(self, "t_fast", t_fast)
        object.__setattr__(self, "beamwidth_rad", beamwidth_rad)
        object.__setattr__(self, "range_pixel_m", range_pixel_m)
        object.__setattr__(self, "azimuth_pixel_m", azimuth_pixel_m)

        if self.SNR_single is not None:
            SNR_lin = 10 ** (self.SNR_single / 10)
            sigma = 1.0  # Reference RCS = 1 m² (0 dBsm)
            Gt_lin = 10 ** (self.transmit_gain_db / 10)
            Gr_lin = 10 ** (self.receive_gain_db / 10)
            N0 = k * self.system_temperature_K * (10 ** (self.noise_figure_db / 10))
            R = self.range_grp_m
            tau = self.pulse_width_sec
            four_pi_cubed = (4 * jnp.pi) ** 3
            Pt = SNR_lin * four_pi_cubed * (R ** 4) * N0 / (Gt_lin * Gr_lin * (lam ** 2) * sigma * tau)
            object.__setattr__(self, "transmit_power_watts", Pt)

        if self.SNR_SAR is not None:
            SNR_lin = 10 ** (self.SNR_SAR / 10)  # linear SAR SNR
            sigma = 1.0  # reference RCS = 1 m²
            Gt_lin = 10 ** (self.transmit_gain_db / 10)
            Gr_lin = 10 ** (self.receive_gain_db / 10)
            R = self.range_grp_m

            # SAR-specific factors from Eq. (1.3)
            denom = (4 * ((4 * jnp.pi)** 3) * R ** 3 * k * self.system_temperature_K *
                     (10 ** (self.noise_figure_db / 10)) * self.bandwidth_hz * self.platform_speed_mps)

            Pave = SNR_lin * denom / (Gt_lin * Gr_lin * (lam ** 3) * sigma * c)

            Pt = Pave / (self.prf_hz * self.pulse_width_sec)

            object.__setattr__(self, "transmit_power_watts", Pt)




@dataclass(frozen=True)
class TargetParams:
    positions_m: jnp.ndarray       # shape (3, N_targets)
    velocities_mps: jnp.ndarray    # shape (3, N_targets)
    accelerations_mps2: jnp.ndarray # shape (3, N_targets)
    rcs_dbsm: jnp.ndarray          # shape (N_targets,)
    phase_rad: jnp.ndarray = None  # shape (N_targets,), optional