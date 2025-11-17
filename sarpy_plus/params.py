# params.py
from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from sarpy_plus import c, k


# ---------------------------------------------------------------------
# Radar parameters
# ---------------------------------------------------------------------

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
    SNR_single: float | None = None
    SNR_SAR: float | None = None
    noise: bool = True
    antenna_pattern: str = "sinc"
    demodulation: Literal["Quadrature", "Dechirp"] = "Quadrature"

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

        # Optional: adjust transmit_power_watts to meet desired SNR_single
        if self.SNR_single is not None:
            SNR_lin = 10 ** (self.SNR_single / 10)
            sigma = 1.0  # Reference RCS = 1 m² (0 dBsm)
            Gt_lin = 10 ** (self.transmit_gain_db / 10)
            Gr_lin = 10 ** (self.receive_gain_db / 10)
            N0 = k * self.system_temperature_K * (10 ** (self.noise_figure_db / 10))
            R = self.range_grp_m
            tau = self.pulse_width_sec
            four_pi_cubed = (4 * jnp.pi) ** 3
            Pt = (
                SNR_lin
                * four_pi_cubed
                * (R**4)
                * N0
                / (Gt_lin * Gr_lin * (lam**2) * sigma * tau)
            )
            object.__setattr__(self, "transmit_power_watts", Pt)

        # Optional: adjust transmit_power_watts to meet desired SNR_SAR
        if self.SNR_SAR is not None:
            SNR_lin = 10 ** (self.SNR_SAR / 10)  # linear SAR SNR
            sigma = 1.0  # reference RCS = 1 m²
            Gt_lin = 10 ** (self.transmit_gain_db / 10)
            Gr_lin = 10 ** (self.receive_gain_db / 10)
            R = self.range_grp_m

            # SAR-specific factors from Eq. (1.3)
            denom = (
                4
                * ((4 * jnp.pi) ** 3)
                * R**3
                * k
                * self.system_temperature_K
                * (10 ** (self.noise_figure_db / 10))
                * self.bandwidth_hz
                * self.platform_speed_mps
            )

            Pave = SNR_lin * denom / (Gt_lin * Gr_lin * (lam**3) * sigma * c)
            Pt = Pave / (self.prf_hz * self.pulse_width_sec)

            object.__setattr__(self, "transmit_power_watts", Pt)

    def __hash__(self) -> int:
        """
        Custom hash so JAX can treat RadarParams as a static argument under jit.
        We intentionally do NOT include array fields (t_fast, t_slow, etc.).
        """
        return hash(
            (
                self.platform_altitude_m,
                self.platform_speed_mps,
                self.center_frequency_hz,
                self.range_resolution_m,
                self.pulse_width_sec,
                self.prf_hz,
                self.cross_range_resolution_m,
                self.ground_range_swath_m,
                self.range_grp_m,
                self.azimuth_aperture_factor,
                self.range_oversample,
                self.noise_power_db,
                float(self.transmit_power_watts),
                self.transmit_gain_db,
                self.receive_gain_db,
                self.system_temperature_K,
                self.noise_figure_db,
                self.SNR_single,
                self.SNR_SAR,
                self.noise,
                self.antenna_pattern,
                self.demodulation,
            )
        )


# ---------------------------------------------------------------------
# Target parameters (pytree)
# ---------------------------------------------------------------------

@register_pytree_node_class
@dataclass(frozen=True)
class TargetParams:
    positions_m: jnp.ndarray         # shape (3, N_targets)
    velocities_mps: jnp.ndarray      # shape (3, N_targets)
    accelerations_mps2: jnp.ndarray  # shape (3, N_targets)
    rcs_dbsm: jnp.ndarray            # shape (N_targets,)
    phase_rad: jnp.ndarray | None = None  # shape (N_targets,), optional

    # JAX pytree methods
    def tree_flatten(self):
        # All the arrays are dynamic children; no auxiliary static data.
        children = (
            self.positions_m,
            self.velocities_mps,
            self.accelerations_mps2,
            self.rcs_dbsm,
            self.phase_rad,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        positions_m, velocities_mps, accelerations_mps2, rcs_dbsm, phase_rad = children
        return cls(
            positions_m=positions_m,
            velocities_mps=velocities_mps,
            accelerations_mps2=accelerations_mps2,
            rcs_dbsm=rcs_dbsm,
            phase_rad=phase_rad,
        )


Pol = Literal["HH", "VV", "HV", "VH"]


# ---------------------------------------------------------------------
# RCS model parameters (static config; hashable)
# ---------------------------------------------------------------------

@dataclass
class RCSParams:
    # Facet (specular-ish) term: Cf * (A_share/λ)^2 * cos^m
    C_f: float = 4.0 * jnp.pi  # scale (tune on a plate)
    m_spec: float = 3.0        # specular sharpness (2–4 common)

    # Edge (wedge-ish) term: Ce * (L_share / sqrt(λ))
    C_e: float = 0.6               # scale (tune on long straight edge)
    wedge_gain_const: float = 1.0  # simple constant; replace with table if desired

    # Corner (multi-bounce glint) term: Cc * (a^4 / λ^2)
    C_c: float = 0.25              # scale (tune on dihedral/trihedral)
    corner_cone_deg: float = 22.5  # optional visibility cone half-angle
    corner_cone_power: float = 4.0 # smooth falloff exponent

    # Small-feature blend: w_O = η^p/(1+η^p),  η=L/λ
    blend_p: float = 2.0
    enable_blend: bool = True

    # Roughness (facet peak reduction): exp(-beta * rough^2)
    rough_beta: float = 4.0
    enable_roughness: bool = True

    # Polarization scalars (keep simple for Level-1)
    pol_facet: dict | None = None
    pol_edge: dict | None = None
    pol_corner: dict | None = None

    sigma_floor: float = 1e-8  # prevents -inf dB

    def __post_init__(self):
        if self.pol_facet is None:
            self.pol_facet = {"HH": 1.0, "VV": 1.0, "HV": 0.1, "VH": 0.1}
        if self.pol_edge is None:
            self.pol_edge = {"HH": 1.0, "VV": 0.8, "HV": 0.2, "VH": 0.2}
        if self.pol_corner is None:
            self.pol_corner = {"HH": 1.0, "VV": 1.0, "HV": 0.3, "VH": 0.3}

    def __hash__(self) -> int:
        # Treat as static config for JIT: hash scalar fields + pol dict contents
        return hash(
            (
                float(self.C_f),
                float(self.m_spec),
                float(self.C_e),
                float(self.wedge_gain_const),
                float(self.C_c),
                float(self.corner_cone_deg),
                float(self.corner_cone_power),
                float(self.blend_p),
                bool(self.enable_blend),
                float(self.rough_beta),
                bool(self.enable_roughness),
                frozenset(self.pol_facet.items()) if self.pol_facet is not None else None,
                frozenset(self.pol_edge.items()) if self.pol_edge is not None else None,
                frozenset(self.pol_corner.items()) if self.pol_corner is not None else None,
                float(self.sigma_floor),
            )
        )


# ---------------------------------------------------------------------
# Scatterer metadata for RCS model (pytree)
# ---------------------------------------------------------------------

@register_pytree_node_class
@dataclass
class ScattererMeta:
    """
    Per-scatterer static metadata.
    All arrays should be shape (S,) except normals/bisectors which are (S,3).
    Fields not used by a given 'kind' can be zeros.
    """
    # kind: 0=facet, 1=edge, 2=corner  (int is JAX-friendly)
    kind: jnp.ndarray            # (S,) int32 in {0,1,2}

    # facet fields
    face_normal: jnp.ndarray     # (S,3) unit vectors; for non-facets leave zeros
    area_share: jnp.ndarray      # (S,) effective area assigned to this dot (m^2)
    roughness: jnp.ndarray       # (S,) surface roughness proxy [0..1]

    # edge fields
    edge_len_share: jnp.ndarray  # (S,) effective length share for edge dots (m)
    wedge_alpha: jnp.ndarray     # (S,) dihedral angle in radians (optional; can be ~π for hard edge)

    # corner fields
    corner_size_a: jnp.ndarray   # (S,) characteristic size a (m) from adjacent edges
    corner_bisector: jnp.ndarray # (S,3) unit vector; if unknown, leave zeros (disables cone gating)

    # JAX pytree methods
    def tree_flatten(self):
        children = (
            self.kind,
            self.face_normal,
            self.area_share,
            self.roughness,
            self.edge_len_share,
            self.wedge_alpha,
            self.corner_size_a,
            self.corner_bisector,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            kind,
            face_normal,
            area_share,
            roughness,
            edge_len_share,
            wedge_alpha,
            corner_size_a,
            corner_bisector,
        ) = children
        return cls(
            kind=kind,
            face_normal=face_normal,
            area_share=area_share,
            roughness=roughness,
            edge_len_share=edge_len_share,
            wedge_alpha=wedge_alpha,
            corner_size_a=corner_size_a,
            corner_bisector=corner_bisector,
        )


# ---------------------------------------------------------------------
# RCS weight computation
# ---------------------------------------------------------------------

def compute_rcs_weights(
    positions_m: jnp.ndarray,   # (3,S)
    sensor_pos_m: jnp.ndarray,  # (Np,3)
    fc_hz: float,
    meta: ScattererMeta,
    params: RCSParams,
    pol: Pol = "HH",
    c: float = 299_792_458.0,
) -> jnp.ndarray:
    """
    Returns σ_lin per scatterer per pulse: (S, Np).
    Level-1 model: facet (projected area * cos^m) + edge (length/sqrt(λ))
    + corner (a^4/λ^2 with optional cone).
    """
    S = positions_m.shape[1]
    Np = sensor_pos_m.shape[0]
    lam = c / fc_hz

    # unit line-of-sight from scatterer TO sensor (so -û is incidence onto surface)
    p = positions_m.T[None, :, :]  # (1,S,3)
    s = sensor_pos_m[:, None, :]   # (Np,1,3)
    u_hat = (s - p) / jnp.linalg.norm(s - p, axis=2, keepdims=True)  # (Np,S,3)

    # Expand metadata to (Np,S,...) where needed
    kind = meta.kind[None, :]              # (1,S)
    n_hat = meta.face_normal[None, :, :]   # (1,S,3)
    area_share = meta.area_share[None, :]  # (1,S)
    rough = meta.roughness[None, :]        # (1,S)

    L_share = meta.edge_len_share[None, :]   # (1,S)
    wedge_alpha = meta.wedge_alpha[None, :]  # (1,S)  # currently unused in Level-1

    a_corner = meta.corner_size_a[None, :]   # (1,S)
    b_hat = meta.corner_bisector[None, :, :] # (1,S,3)

    # Kind masks
    is_f = (kind == 0)
    is_e = (kind == 1)
    is_c = (kind == 2)

    # ---------- Facet term ----------
    # cos_inc = n · (-û); clamp [0,1]
    cos_inc = jnp.clip(jnp.sum(n_hat * (-u_hat), axis=2), 0.0, 1.0)  # (Np,S)
    facet_base = params.C_f * ((area_share / lam) ** 2) * (cos_inc ** params.m_spec)  # (Np,S)

    if params.enable_roughness:
        facet_base = facet_base * jnp.exp(-params.rough_beta * (rough ** 2))

    facet_pol = params.pol_facet.get(pol, 1.0)
    facet_sigma = facet_base * facet_pol * is_f  # mask to facets

    # ---------- Edge term ----------
    # Simple constant wedge factor for Level-1; you can make this function of (alpha, ψ)
    edge_sigma = params.C_e * (L_share / jnp.sqrt(lam)) * params.wedge_gain_const  # (1,S)
    edge_sigma = jnp.broadcast_to(edge_sigma, (Np, S)) * is_e

    # ---------- Corner term ----------
    corner_base = params.C_c * ((a_corner ** 4) / (lam ** 2))  # (1,S)
    if params.corner_cone_deg is not None and params.corner_cone_deg > 0.0:
        # Angle between view direction and bisector; if bisector is zero, gate=1
        b_norm = jnp.linalg.norm(b_hat, axis=2, keepdims=True)
        has_b = (b_norm[..., 0] > 0.0)  # (1,S)
        b_unit = jnp.where(b_norm > 0.0, b_hat / b_norm, b_hat)  # (1,S,3)
        cos_phi = jnp.clip(jnp.sum(b_unit * u_hat, axis=2), -1.0, 1.0)  # (Np,S)
        phi = jnp.arccos(cos_phi)
        phi0 = jnp.deg2rad(params.corner_cone_deg)
        # smooth cone gate: cos^p inside cone, drops outside
        h = jnp.where(
            phi <= phi0,
            jnp.cos(jnp.pi * 0.5 * (phi / phi0)) ** params.corner_cone_power,
            0.0,
        )
        # if no bisector, default to always on
        h = jnp.where(jnp.broadcast_to(has_b, h.shape), h, 1.0)
    else:
        h = jnp.ones((Np, S))

    corner_pol = params.pol_corner.get(pol, 1.0)
    corner_sigma = jnp.broadcast_to(corner_base, (Np, S)) * h * corner_pol * is_c

    # ---------- Blend for small features (optional) ----------
    if params.enable_blend:
        # Characteristic size L per dot: sqrt(A) for facets, L_share for edges, a for corners
        L_f = jnp.sqrt(jnp.clip(area_share, 0.0, jnp.inf))
        L_e = jnp.clip(L_share, 0.0, jnp.inf)
        L_c = jnp.clip(a_corner, 0.0, jnp.inf)

        # Select L by kind
        L = (
            L_f * is_f
            + (jnp.broadcast_to(L_e, (Np, S)) * is_e)
            + (jnp.broadcast_to(L_c, (Np, S)) * is_c)
        )
        eta = L / lam
        wO = (eta ** params.blend_p) / (1.0 + eta ** params.blend_p)  # optical weight
        # push facets a bit more toward optical at high cos_inc (helps look)
        wO = jnp.where(is_f, wO * (0.5 + 0.5 * cos_inc), wO)
    else:
        wO = jnp.ones((Np, S))

    # Summation and floor
    sigma = wO * facet_sigma + wO * edge_sigma + wO * corner_sigma
    sigma = jnp.clip(sigma, params.sigma_floor, jnp.inf)  # (Np,S)

    # Return as (S,Np) to match your sim style (scatterers x pulses)
    return sigma.T
