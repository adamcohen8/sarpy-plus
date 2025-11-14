import jax.numpy as jnp
import jax
from sarpy_plus.params import (RadarParams,
                               TargetParams,
                               compute_rcs_weights,
                               RCSParams,
                               ScattererMeta)
from sarpy_plus.constants import  c, k
from dataclasses import dataclass
import numpy as np


def ray_triangle_mt(orig: np.ndarray, dir: np.ndarray,
                    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                    tmax: float, eps: float = 1e-9) -> bool:
    """
    Möller–Trumbore intersection: return True if segment [orig, orig+dir*tmax] hits triangle
    strictly before tmax (0 < t < tmax).
    """
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(dir, e2)
    det  = float(np.dot(e1, pvec))
    if abs(det) < eps:
        return False
    inv_det = 1.0 / det
    tvec = orig - v0
    u = float(np.dot(tvec, pvec)) * inv_det
    if u < 0.0 or u > 1.0:
        return False
    qvec = np.cross(tvec, e1)
    v = float(np.dot(dir, qvec)) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return False
    t = float(np.dot(e2, qvec)) * inv_det
    if t <= eps or t >= tmax - eps:
        return False
    return True

@dataclass
class BVHNode:
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    left:  int
    right: int
    start: int
    count: int

class BVH:
    """
    Simple midpoint-split BVH over triangles for fast segment-occlusion queries.
    verts: (V,3) float, faces: (F,3) int
    """
    def __init__(self, verts: np.ndarray, faces: np.ndarray, leaf_size: int = 8):
        self.verts = np.asarray(verts, float)
        self.faces = np.asarray(faces, int)
        self.leaf_size = int(leaf_size)

        self.tri_v0 = self.verts[self.faces[:, 0]]
        self.tri_v1 = self.verts[self.faces[:, 1]]
        self.tri_v2 = self.verts[self.faces[:, 2]]

        tri_min = np.minimum(np.minimum(self.tri_v0, self.tri_v1), self.tri_v2)
        tri_max = np.maximum(np.maximum(self.tri_v0, self.tri_v1), self.tri_v2)
        self.tri_min = tri_min
        self.tri_max = tri_max
        self.tri_cent = (tri_min + tri_max) * 0.5

        self.index = np.arange(self.faces.shape[0], dtype=int)
        self.nodes: list[BVHNode] = []
        self._build(0, self.faces.shape[0])

    @staticmethod
    def from_mesh(verts: np.ndarray, faces: np.ndarray, leaf_size: int = 8) -> "BVH":
        return BVH(verts, faces, leaf_size)

    def _make_node(self, imin, imax) -> int:
        tri_ids = self.index[imin:imax]
        bmin = np.min(self.tri_min[tri_ids], axis=0)
        bmax = np.max(self.tri_max[tri_ids], axis=0)
        node_id = len(self.nodes)
        self.nodes.append(BVHNode(bmin, bmax, -1, -1, imin, imax - imin))
        return node_id

    def _build(self, imin, imax) -> int:
        node_id = self._make_node(imin, imax)
        count = imax - imin
        if count <= self.leaf_size:
            return node_id
        tri_ids = self.index[imin:imax]
        ext = self.nodes[node_id].bbox_max - self.nodes[node_id].bbox_min
        axis = int(np.argmax(ext))
        if ext[axis] <= 0:
            return node_id
        cent = self.tri_cent[tri_ids, axis]
        mid = np.median(cent)
        left_mask = cent <= mid
        nleft = int(np.count_nonzero(left_mask))
        if nleft == 0 or nleft == count:
            # degenerate split → sort & split in half
            order = np.argsort(cent)
            self.index[imin:imax] = tri_ids[order]
            nleft = count // 2
        else:
            self.index[imin:imax] = np.concatenate([tri_ids[left_mask], tri_ids[~left_mask]])
        left_id  = self._build(imin, imin + nleft)
        right_id = self._build(imin + nleft, imax)
        self.nodes[node_id].left  = left_id
        self.nodes[node_id].right = right_id
        return node_id

    @staticmethod
    def _aabb_hit(orig, inv_dir, sign, bmin, bmax, tmax):
        # Robust slab test with precomputed inv_dir and sign bits
        tmin = ((bmin[0] if sign[0] else bmax[0]) - orig[0]) * inv_dir[0]
        tmax2= ((bmax[0] if sign[0] else bmin[0]) - orig[0]) * inv_dir[0]
        tymin= ((bmin[1] if sign[1] else bmax[1]) - orig[1]) * inv_dir[1]
        tymax= ((bmax[1] if sign[1] else bmin[1]) - orig[1]) * inv_dir[1]
        if tmin > tymax or tymin > tmax2:
            return False
        if tymin > tmin: tmin = tymin
        if tymax < tmax2: tmax2 = tymax
        tzmin= ((bmin[2] if sign[2] else bmax[2]) - orig[2]) * inv_dir[2]
        tzmax= ((bmax[2] if sign[2] else bmin[2]) - orig[2]) * inv_dir[2]
        if tmin > tzmax or tzmin > tmax2:
            return False
        if tzmin > tmin: tmin = tzmin
        if tzmax < tmax2: tmax2 = tzmax
        return (tmin < tmax2) and (tmin < tmax) and (tmax2 > 0.0)

    def occluded_segment(self, orig: np.ndarray, end: np.ndarray, eps: float = 1e-9) -> bool:
        """
        True if any triangle intersects the open segment (orig, end).
        """
        dir = end - orig
        tmax = float(np.linalg.norm(dir))
        if tmax <= eps:
            return False
        dir /= tmax
        inv_dir = 1.0 / np.where(np.abs(dir) < eps, np.sign(dir)*eps, dir)
        sign = (inv_dir < 0.0)

        stack = [0]  # root
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            if not self._aabb_hit(orig, inv_dir, sign, node.bbox_min, node.bbox_max, tmax):
                continue
            if node.count <= self.leaf_size:
                tri_ids = self.index[node.start:node.start + node.count]
                for tid in tri_ids:
                    if ray_triangle_mt(orig, dir,
                                       self.tri_v0[tid], self.tri_v1[tid], self.tri_v2[tid],
                                       tmax, eps):
                        return True
            else:
                if node.left  != -1: stack.append(node.left)
                if node.right != -1: stack.append(node.right)
        return False

    def visible_mask_segment(self, sensor_o: np.ndarray, pts: np.ndarray,
                             eps_pullback: float, eps_ray: float) -> np.ndarray:
        """
        pts: (N,3). Returns bool[N] where True means visible from sensor_o.
        Pulls the endpoint slightly toward the sensor to avoid self-hit.
        """
        N = pts.shape[0]
        vis = np.zeros(N, dtype=bool)
        for i in range(N):
            p = pts[i]
            d = p - sensor_o
            L = float(np.linalg.norm(d))
            if L <= eps_ray:
                vis[i] = False
                continue
            p_back = p - (eps_pullback * d / L)
            vis[i] = not self.occluded_segment(sensor_o, p_back, eps=eps_ray)
        return vis








def SAR_Sim(
    radar: RadarParams,
    tgt: TargetParams,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    *,
    bvh=None,                               # BVH instance or None
    meta: ScattererMeta | None = None,      # per-scatterer metadata for RCS model
    rcs_params: RCSParams | None = None,    # optional RCS tuning params
    vis_eps_pullback_frac: float = 1e-4,    # for BVH visibility
    vis_eps_ray_frac: float = 1e-6
) -> jnp.ndarray:
    """
    High-fidelity SAR raw phase-history simulator (complex voltage).

    Behavior:
      - If bvh is None:
          * Identical to original SAR_Sim:
              - Uses tgt.rcs_dbsm as before (no aspect/pol RCS model)
              - No occlusion culling
      - If bvh is not None:
          * Applies BVH-based self-occlusion per pulse (visibility mask)
          * If meta is not None:
                uses compute_rcs_weights(...) for σ_lin per scatterer & pulse
            else:
                falls back to tgt.rcs_dbsm (old RCS model), but WITH occlusion
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
    mid_pos = jnp.array([
        -radar.range_grp_m * jnp.cos(grazing),
        0.0,
        radar.platform_altitude_m
    ])

    sar_pos = jnp.stack(
        (jnp.full(Np, mid_pos[0]),
         t_slow * radar.platform_speed_mps,
         jnp.full(Np, mid_pos[2])), axis=0
    )  # (3, Np)

    # ---------- Target state ----------
    Ntgt    = tgt.rcs_dbsm.size
    phase0  = (tgt.phase_rad
               if tgt.phase_rad is not None else jnp.zeros_like(tgt.rcs_dbsm))
    t_slow_b = t_slow[None, :]

    r0 = (tgt.positions_m[:, :, None] +
          tgt.velocities_mps[:, :, None] * t_slow_b +
          0.5 * tgt.accelerations_mps2[:, :, None] * t_slow_b ** 2)  # (3,Ntgt,Np)

    rel = r0 - sar_pos[:, None, :]              # (3, Ntgt, Np)
    R   = jnp.linalg.norm(rel, axis=0)          # (Ntgt, Np)

    # ---------- Antenna pattern ----------
    beam_dir = jnp.array([jnp.cos(grazing), 0., -jnp.sin(grazing)])
    theta    = jnp.arccos(
        jnp.einsum("ijk,i->jk", rel, beam_dir) /
        (R * jnp.linalg.norm(beam_dir))
    )
    theta_3dB = radar.beamwidth_rad / 2.0

    if radar.antenna_pattern == "binary":
        G_beam_v = (theta <= theta_3dB).astype(jnp.float32)
    elif radar.antenna_pattern == "parabolic":
        u = jnp.pi * theta / theta_3dB
        pattern = jnp.where(u < 1e-6, 1.0,
                            3 * (jnp.sin(u) - u * jnp.cos(u)) / u ** 3)
        G_beam_v = jnp.abs(pattern)
    elif radar.antenna_pattern == "gaussian":
        G_beam_v = jnp.exp(- (jnp.log(2) / 2) * (theta / theta_3dB) ** 2)
    elif radar.antenna_pattern == "sinc":  # default to "sinc"
        G_beam_v = jnp.sqrt((jnp.sinc(theta / theta_3dB)) ** 2)
    else:  # Spotlight Mode (no pattern)
        G_beam_v = 1.0

    # ---------- Voltage-range equation ----------
    Gt_lin  = 10.0 ** (radar.transmit_gain_db / 10.0)
    Gr_lin  = 10.0 ** (radar.receive_gain_db / 10.0)

    A0 = (jnp.sqrt(radar.transmit_power_watts * Gt_lin * Gr_lin) *
          lam_fac)                                            # constant part

    # ---------- Visibility mask via BVH (if any) ----------
    if bvh is not None:
        # Positions & sensor to numpy for BVH traversal
        # r0: (3, Ntgt, Np) → (Ntgt,3,Np)
        r0_np = np.asarray(np.array(r0.transpose(1, 0, 2)))  # (Ntgt,3,Np)
        sar_pos_np = np.asarray(np.array(sar_pos.T))         # (Np,3)

        # Scene extent for epsilons (based on initial positions)
        pts0_np = np.asarray(np.array(tgt.positions_m.T))    # (Ntgt,3)
        if pts0_np.size > 0:
            scene_extent = float(np.max(np.ptp(pts0_np, axis=0)))
        else:
            scene_extent = 1.0
        eps_pull = max(vis_eps_pullback_frac * scene_extent, 1e-9)
        eps_ray  = max(vis_eps_ray_frac       * scene_extent, 1e-9)

        vis_mask = np.zeros((Ntgt, Np), dtype=bool)
        for p in range(Np):
            s_o = sar_pos_np[p]          # (3,)
            pts_p = r0_np[:, :, p]       # (Ntgt,3)
            vis_mask[:, p] = bvh.visible_mask_segment(s_o, pts_p,
                                                      eps_pull, eps_ray)

        vis_mask_j = jnp.asarray(vis_mask)   # (Ntgt, Np)
    else:
        # No BVH: everything is visible (old behavior)
        vis_mask_j = jnp.ones((Ntgt, Np), dtype=bool)

    # ---------- RCS model: conditional on BVH/meta ----------
    # If bvh is None → use original tgt.rcs_dbsm model.
    # If bvh is not None and meta is provided → use new RCS model.
    # If bvh is not None and meta is None → old RCS model but with occlusion.

    if (bvh is not None) and (meta is not None):
        # New Level-1 RCS model (aspect + mechanism based)
        if rcs_params is None:
            rcs_params = RCSParams()

        # sensor positions per pulse: (Np,3)
        sensor_pos_m = sar_pos.T

        # carrier frequency from λ
        fc_hz = float(c / lam)

        pol = getattr(radar, "polarization", "HH")

        sigma_lin = compute_rcs_weights(
            positions_m=tgt.positions_m,      # (3,Ntgt)
            sensor_pos_m=sensor_pos_m,        # (Np,3)
            fc_hz=fc_hz,
            meta=meta,
            params=rcs_params,
            pol=pol,
            c=float(c)
        )  # (Ntgt, Np)

        sigma_v = jnp.sqrt(sigma_lin)        # (Ntgt, Np)

    else:
        # Old behavior: σ from tgt.rcs_dbsm, no aspect dependence
        # Original code had sigma_v shape (Ntgt,1); repeat over pulses.
        sigma_v0 = jnp.sqrt(10.0 ** (tgt.rcs_dbsm / 10.0))[:, None]  # (Ntgt,1)
        sigma_v = jnp.repeat(sigma_v0, Np, axis=1)                   # (Ntgt,Np)

    # Apply beam pattern, range loss, and visibility mask
    A_R = (A0 * sigma_v * G_beam_v * R ** (-2) *
           vis_mask_j.astype(jnp.float32))        # (Ntgt, Np)

    # ---------- Fast-time envelope ----------
    tau  = 2.0 * R / c
    tcen = t_fast[:, None, None] - tau[None, :, :]

    if radar.demodulation == 'Dechirp':
        tau_ref = 2.0 * radar.range_grp_m / c
        mask = jnp.abs(t_fast[:, None, None] - tau_ref) < radar.pulse_width_sec / 2.0
        tcen_ref = t_fast[:, None, None] - tau_ref
        dechirp_p = -jnp.pi * radar.chirp_rate_hz_per_sec * tcen_ref ** 2
    else:  # 'Quadrature'
        mask = jnp.abs(tcen) < radar.pulse_width_sec / 2.0
        dechirp_p = 0.0  # No dechirp

    p1 = -4.0 * jnp.pi * R                # 2-way phase (divide by λ later)
    p2 = jnp.pi * radar.chirp_rate_hz_per_sec * tcen ** 2
    total_phase = (p1 / lam) + phase0[:, None] + p2 + dechirp_p  # (Nr,Ntgt,Np)

    target_v = A_R[None, :, :] * jnp.exp(1j * total_phase) * mask.astype(jnp.float32)
    ph = jnp.sum(target_v, axis=1)                            # (Nr, Np)

    # ---------- kTB noise (complex) ----------
    Tsys  = radar.system_temperature_K
    F_lin = 10.0 ** (radar.noise_figure_db / 10.0)
    N0    = k * Tsys * F_lin              # W / Hz
    Pn    = N0 * fs                       # total noise power in bandwidth
    noise_std = jnp.sqrt(Pn / 2.0)        # per real component

    key, k_r, k_i = jax.random.split(key, 3)
    noise_real = jax.random.normal(k_r, ph.shape) * noise_std
    noise_imag = jax.random.normal(k_i, ph.shape) * noise_std

    if radar.noise:
        ph = ph + (noise_real + 1j * noise_imag)

    return ph
