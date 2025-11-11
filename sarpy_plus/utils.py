import numpy as np
import matplotlib.pyplot as plt
from sarpy_plus.constants import c, k
from sarpy_plus.params import RadarParams
import jax.numpy as jnp
import jax

def plot_space_2d(img: np.ndarray, radar, cmap: str = 'bone', title: str = 'SAR Image | Space Domain', window: float = None, db: bool = False):
    """
    Plot 2D SAR image (absolute value), axes in meters.
    X axis → Azimuth (m), Y axis → Range (m)

    window: if set, zoom to ±window meters around brightest pixel.
    """
    range_vector_m = radar.t_fast * c / 2.0
    az_vector_m = radar.t_slow * radar.platform_speed_mps

    # Determine extent
    extent = [
        az_vector_m[0], az_vector_m[-1],      # X axis → Azimuth
        range_vector_m[-1], range_vector_m[0] # Y axis → Range (flip to have near at top)
    ]

    if window is not None:

        brightest_range_idx, brightest_az_idx = np.unravel_index(np.argmax(np.abs(img)), img.shape)

        # Centers in **METERS**
        range_center_m = range_vector_m[brightest_range_idx]
        az_center_m = az_vector_m[brightest_az_idx]

        # **SLICES: Find start/end indices** (±window m) → **CLIPS EDGES AUTO**
        range_slice = slice(
            np.searchsorted(range_vector_m, range_center_m - window),
            np.searchsorted(range_vector_m, range_center_m + window)
        )
        az_slice = slice(
            np.searchsorted(az_vector_m, az_center_m - window),
            np.searchsorted(az_vector_m, az_center_m + window)
        )

        # **CROP: High-res sub-image**
        img_cropped = img[range_slice, az_slice]
        range_cropped = range_vector_m[range_slice]
        az_cropped = az_vector_m[az_slice]

        # **EXTENT: X=left→right, Y=far(bottom)→near(top) = FLIPPED**
        extent = [
            az_cropped[0], az_cropped[-1],  # X: Az min→max
            range_cropped[-1], range_cropped[0]  # Y: far→near (TOP=near)
        ]

        img = img_cropped

    plt.figure(figsize=(8, 6))
    if db:
        plt.imshow(10*np.log10(np.abs(img)), cmap=cmap, extent=extent)
    else:
        plt.imshow(np.abs(img), aspect='auto', extent=extent, cmap=cmap)
    plt.xlabel('Azimuth (m)')
    plt.ylabel('Range (m)')
    plt.colorbar(label='Amplitude')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_time_2d(ph: np.ndarray, radar, component: str = 'real', cmap: str = 'bone', title: str = 'Phase History | Time Domain'):
    """
    Plot 2D phase history in time domain.
    component: 'real', 'imag', or 'abs'
    """
    if component == 'real':
        data = np.real(ph)
        label = 'Real(Phase History)'
    elif component == 'imag':
        data = np.imag(ph)
        label = 'Imag(Phase History)'
    elif component == 'abs':
        data = np.abs(ph)
        label = 'Abs(Phase History)'
    else:
        raise ValueError("component must be 'real', 'imag', or 'abs'")

    plt.figure(figsize=(8, 6))
    extent = [
        radar.t_slow[0], radar.t_slow[-1],
        radar.t_fast[-1] * 1e6, radar.t_fast[0] * 1e6
    ]
    plt.imshow(data, aspect='auto', extent=extent, cmap=cmap)
    plt.xlabel('Slow Time (s)')
    plt.ylabel('Fast Time (μs)')
    plt.colorbar(label=label)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_space_1d(img: np.ndarray, radar, axis: str = 'range', index: int = None, window: float = None):
    """
    Plot 1D cut of SAR image along range or azimuth.
    If index is None, use brightest column/row.
    If window is set, zoom to ±window meters around brightest pixel.
    """
    range_vector_m = radar.t_fast * c / 2.0
    az_vector_m = radar.t_slow * radar.platform_speed_mps

    if axis == 'range':
        if index is None:
            index = np.argmax(np.sum(np.abs(img), axis=0))
        profile = np.abs(img[:, index])
        x_axis = range_vector_m
        x_label = 'Range (m)'
    elif axis == 'azimuth':
        if index is None:
            index = np.argmax(np.sum(np.abs(img), axis=1))
        profile = np.abs(img[index, :])
        x_axis = az_vector_m
        x_label = 'Azimuth (m)'
    else:
        raise ValueError("axis must be 'range' or 'azimuth'")

    # Determine zoom window if requested
    if window is not None:
        max_idx = np.argmax(profile)
        center = x_axis[max_idx]
        mask = (x_axis >= center - window) & (x_axis <= center + window)
        x_axis = x_axis[mask]
        profile = profile[mask]

    plt.figure(figsize=(8, 4))
    plt.plot(x_axis, profile, color='blue')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')
    plt.title(f'1D Cut | Space Domain | Axis: {axis}, Index: {index}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_time_1d(ph: np.ndarray, radar, axis: str = 'range', index: int = None, component: str = 'real'):
    """
    Plot 1D cut of phase history along fast time (range) or slow time (azimuth).
    If index is None, use brightest column/row.
    component: 'real', 'imag', or 'abs'
    """
    if axis == 'range':
        if index is None:
            index = np.argmax(np.sum(np.abs(ph), axis=0))
        vec = ph[:, index]
        x_axis = radar.t_fast * 1e6
        x_label = 'Fast Time (μs)'
    elif axis == 'azimuth':
        if index is None:
            index = np.argmax(np.sum(np.abs(ph), axis=1))
        vec = ph[index, :]
        x_axis = radar.t_slow
        x_label = 'Slow Time (s)'
    else:
        raise ValueError("axis must be 'range' or 'azimuth'")

    if component == 'real':
        profile = np.real(vec)
        label = 'Real'
    elif component == 'imag':
        profile = np.imag(vec)
        label = 'Imag'
    elif component == 'abs':
        profile = np.abs(vec)
        label = 'Abs'
    else:
        raise ValueError("component must be 'real', 'imag', or 'abs'")

    plt.figure(figsize=(8, 4))
    plt.plot(x_axis, profile, color='blue')
    plt.xlabel(x_label)
    plt.ylabel(label)
    plt.title(f'1D Cut | Time Domain | Axis: {axis}, Index: {index}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calc_snr_single(radar, tgt):
    """
    Single-pulse SNR (dB) for the lowest-RCS target.

    Parameters
    ----------
    radar : RadarParams
        Must include:
          • transmit_power_watts
          • transmit_gain_db
          • receive_gain_db
          • center_wavelength_m
          • range_grp_m
          • sample_rate_hz          (≈ processed bandwidth B)
          • system_temperature_K    (Tsys)
          • noise_figure_db         (F, in dB)
    tgt : TargetParams
        Must include rcs_dbsm (array-like).

    Returns
    -------
    float
        SNR in decibels for a single, uncompressed pulse.
    """
    # ---------- Constants ----------
    lam = radar.center_wavelength_m          # wavelength (m)
    R   = radar.range_grp_m                  # slant range (m)

    # Lowest-RCS target (worst-case SNR)
    sigma = 10.0 ** (jnp.min(tgt.rcs_dbsm) / 10.0)

    # Antenna gains to linear scale
    Gt = 10.0 ** (radar.transmit_gain_db / 10.0)
    Gr = 10.0 ** (radar.receive_gain_db  / 10.0)

    # ---------- Received power (monostatic radar-range equation) ----------
    #   Pr = Pt * Gt * Gr * λ² * σ / ( (4π)³ R⁴ )
    P_r = (radar.transmit_power_watts * Gt * Gr *
           lam**2 * sigma) / ((4.0 * jnp.pi)**3 * R**4)

    # ---------- Noise power (k Tsys F B) ----------
    Tsys  = radar.system_temperature_K
    F_lin = 10.0 ** (radar.noise_figure_db / 10.0)
    B     = radar.sample_rate_hz              # bandwidth ≈ sample rate
    P_n   = k * Tsys * F_lin * B            # Watts

    # ---------- SNR ----------
    snr_linear = P_r / P_n
    snr_db = 10.0 * jnp.log10(snr_linear + 1e-30)  # protect log(0)

    return float(snr_db)


def nextpow2(x: int) -> int:
    """
    Return the smallest power‐of‐two >= x.
    """
    x = int(x)
    if x <= 1:
        return 1
    # bit_length trick: 1 << (n-1).bit_length() is the next pow2
    return 1 << (x - 1).bit_length()




def integer_sequence(N: int, centered: bool = False) -> np.ndarray:
    """
    dsp.integer_sequence(N, centered)

    — If centered=False, returns [0,1,2,...,N-1].
    — If centered=True, returns a length‑N sequence of integers (or floats)
      centered about zero, i.e. from −(N-1)/2 to +(N-1)/2 in steps of 1.

    Examples:
      integer_sequence(5, False) -> array([0, 1, 2, 3, 4])
      integer_sequence(5, True)  -> array([-2., -1.,  0.,  1.,  2.])
      integer_sequence(4, True)  -> array([-1.5, -0.5,  0.5,  1.5])
    """
    if not centered:
        return np.arange(N)
    # centered about zero:
    return np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)


def column_vector(x: np.ndarray) -> np.ndarray:
    """
    fmt.column_vector(x)

    Take a 1‑D array of shape (N,) and make it into a column vector (N,1).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("column_vector: input must be 1-D")
    return x.reshape(-1, 1)



def plot_range_doppler_map(ph: np.ndarray,
                           radar: RadarParams,
                           n_fft: int = None,
                           window: np.ndarray = None,
                           az_axis: np.ndarray = None,
                           rg_axis: np.ndarray = None,
                           vmin: float = -40,
                           vmax: float = 0,
                           cmap: str = 'bone'):
    """
    Plot Range-Doppler map (dB scale) from phase history data.

    Parameters
    ----------
    ph : np.ndarray, shape (n_ranges, n_pulses)
        Phase history (fast time vs. slow time).
    radar : RadarParams
        Radar parameters, used for PRF and optional axis computation.
    n_fft : int, optional
        Number of points for azimuth FFT (defaults to n_pulses).
    window : np.ndarray, optional
        1-D window of length n_pulses to apply before FFT.
    az_axis : np.ndarray, optional
        Azimuth (Doppler) axis values. Defaults to Doppler frequencies (Hz).
    rg_axis : np.ndarray, optional
        Range axis values. Defaults to range bins (indices).
    vmin, vmax : float, optional
        dB display limits (default -40 to 0 dB).
    cmap : str, optional
        Matplotlib colormap (default 'bone').
    """
    n_ranges, n_pulses = ph.shape
    M = n_fft or n_pulses

    # Apply window if provided
    if window is not None:
        ph = ph * window[None, :]

    # FFT along pulses → Range–Doppler, then shift
    rd = np.fft.fft(ph, n=M, axis=1)
    rd = np.fft.fftshift(rd, axes=1)

    # Convert to dB and normalize
    S = np.abs(rd)
    S_db = 20 * np.log10(S / np.max(S + 1e-30))

    # Doppler axis
    fd = np.fft.fftshift(np.fft.fftfreq(M, d=1.0 / radar.prf_hz))

    # Default axes if not provided
    if az_axis is None:
        az_axis = fd
        x_label = 'Doppler (Hz)'
    else:
        x_label = 'Azimuth'  # Generic if custom
    if rg_axis is None:
        rg_axis = np.arange(n_ranges)
        y_label = 'Range bin'
        do_invert = False
    else:
        y_label = 'Range'
        do_invert = False  # Assume user handles orientation

    # If using default range, option to use meters (but keep as bins by default to match provided)
    # For consistency with other plots, could add a flag later if needed.

    # Plot
    plt.figure(figsize=(8, 5))
    pcm = plt.pcolormesh(az_axis, rg_axis, S_db, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(pcm, label='Amplitude (dB)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Range–Doppler Map')
    if do_invert:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def PSLR(image):
    """
    Compute Peak Sidelobe Ratio (PSLR) in dB for both range and azimuth.

    Args:
        image (jnp.ndarray): 2D complex SAR image.
        radar (RadarParams): Radar parameter object from params.py.

    Returns:
        tuple: (PSLR_range_dB, PSLR_azimuth_dB)
    """

    # Extract 1D cuts
    index = jnp.argmax(jnp.sum(jnp.abs(image), axis=0))
    rg_cut = jnp.abs(image[:, index])**2

    index = jnp.argmax(jnp.sum(jnp.abs(image), axis=1))
    az_cut = jnp.abs(image[index, :])**2

    def compute_pslr(cut):
        peak_val = jnp.max(cut)
        peak_idx = jnp.argmax(cut)

        profile = cut / jnp.max(cut)
        above_half = jnp.where(profile >= 0.1)[0]
        peak_idx = jnp.argmax(cut)
        above_half = contiguous_region(above_half, peak_idx)
        if len(above_half) < 2:
            return 0.0
        width_px = above_half[-1] - above_half[0]


        # Exclude ±1 pixels around main lobe
        sidelobes = jnp.concatenate([cut[:jnp.maximum(0, peak_idx - width_px)], cut[peak_idx + width_px:]])
        sidelobe_peak = jnp.max(sidelobes)
        return 10.0 * jnp.log10(peak_val / sidelobe_peak)

    pslr_rg_dB = compute_pslr(rg_cut)
    pslr_az_dB = compute_pslr(az_cut)

    return pslr_rg_dB, pslr_az_dB


def IRW(image, radar: RadarParams):
    """
    Compute Impulse Response Width (-3 dB width) in meters for range and azimuth.

    Args:
        image (jnp.ndarray): 2D complex SAR image.
        radar (RadarParams): Radar parameter object from params.py.

    Returns:
        tuple: (IRW_range_m, IRW_azimuth_m)
    """
    # Extract the 1D profiles
    index = jnp.argmax(jnp.sum(jnp.abs(image), axis=0))
    rg_cut = jnp.abs(image[:, index])

    index = jnp.argmax(jnp.sum(jnp.abs(image), axis=1))
    az_cut = jnp.abs(image[index, :])

    def width_at_half_power(cut, resolution):
        cut = cut / jnp.max(cut)
        above_half = jnp.where(cut >= 0.5)[0]
        peak_idx = jnp.argmax(cut)
        above_half = contiguous_region(above_half, peak_idx)
        if len(above_half) < 2:
            return 0.0
        width_px = above_half[-1] - above_half[0]
        return width_px * resolution

    irw_rg_m = width_at_half_power(rg_cut, radar.range_pixel_m)
    irw_az_m = width_at_half_power(az_cut, radar.azimuth_pixel_m)

    return irw_rg_m, irw_az_m


def ISLR(image, alpha:float=1.0):

    # -------- Range (rg) --------
    index = jnp.argmax(jnp.sum(jnp.abs(image), axis=0))
    rg_cut = jnp.abs(image[:, index])

    cut = rg_cut / jnp.max(rg_cut)
    above_half_rg = jnp.where(cut >= 0.5)[0]
    peak_rg = jnp.argmax(cut)

    above_half_rg = contiguous_region(above_half_rg, peak_rg)


    # -------- Azimuth (az) --------
    index = jnp.argmax(jnp.sum(jnp.abs(image), axis=1))
    az_cut = jnp.abs(image[index, :])

    cut = az_cut / jnp.max(az_cut)
    above_half_az = jnp.where(cut >= 0.5)[0]
    peak_az = jnp.argmax(cut)


    above_half_az = contiguous_region(above_half_az, peak_az)


    # -------- ISLR calculation --------
    rg_pwr = rg_cut ** 2
    az_pwr = az_cut ** 2


    islr_rg = 10 * jnp.log10(
        (jnp.sum(rg_pwr) - alpha*jnp.sum(rg_pwr[above_half_rg])) / (alpha*jnp.sum(rg_pwr[above_half_rg]))
    )
    islr_az = 10 * jnp.log10(
        (jnp.sum(az_pwr) - alpha*jnp.sum(az_pwr[above_half_az])) / (alpha*jnp.sum(az_pwr[above_half_az]))
    )

    return islr_rg, islr_az


def contiguous_region(above_half, peak_idx):
    """Return only the contiguous subset of above_half that contains peak_idx."""
    # Identify breaks between consecutive indices
    diffs = jnp.diff(above_half)
    # Get start/end indices of contiguous regions
    split_points = jnp.where(diffs > 1)[0] + 1
    groups = jnp.split(above_half, split_points)

    # Find which group contains the peak index
    for g in groups:
        if peak_idx in g:
            return g
    return above_half  # fallback (shouldn't happen)