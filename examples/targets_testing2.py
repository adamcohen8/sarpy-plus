
# ------------------------------------------------------------
# Scatterer generator: "polkadot skin"
#   • orientation (auto/manual)
#   • optional subdivision (mid-edge)
#   • silhouette edge boost (view-independent)
#   • octant balancing (avoids “half missing”)
#   • edge-hugging surface dots, min-per-face, tiny jitter
# ------------------------------------------------------------

from typing import Tuple, Optional
import numpy as np
import jax.numpy as jnp
from sarpy_plus.params import TargetParams
from sarpy_plus.targets import plot_scatterers_3d



# ---------- OBJ parsing (v/f only, fan-triangulate) ----------
def parse_obj_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            if line.startswith('v '):
                parts = line.strip().split()[1:]
                if len(parts) < 3:
                    raise ValueError(f"Invalid vertex: {line}")
                verts.append([float(p) for p in parts[:3]])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                idxs = [int(p.split('/')[0]) - 1 for p in parts]
                if len(idxs) < 3:
                    raise ValueError(f"Invalid face: {line}")
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if verts.size == 0 or faces.size == 0:
        raise ValueError("OBJ has no geometry.")
    if verts.shape[1] != 3:
        raise ValueError(f"Verts not 3D: {verts.shape}")
    return verts, faces

# ---------- Geometry helpers ----------
def _face_geom(verts: np.ndarray, faces: np.ndarray):
    a = verts[faces[:, 0]]; b = verts[faces[:, 1]]; c = verts[faces[:, 2]]
    ab = b - a; ac = c - a
    normals = np.cross(ab, ac)
    area2 = np.linalg.norm(normals, axis=1)       # = 2*area
    areas = 0.5 * area2
    normals = normals / np.clip(area2[:, None], 1e-12, None)
    perims = (np.linalg.norm(ab, axis=1) +
              np.linalg.norm(c - b, axis=1) +
              np.linalg.norm(a - c, axis=1))
    return normals, areas, perims

def _edge_map(faces: np.ndarray):
    m = {}
    for fi, f in enumerate(faces):
        e = [(int(f[0]),int(f[1])), (int(f[1]),int(f[2])), (int(f[2]),int(f[0]))]
        for i,j in e:
            k = (i,j) if i<j else (j,i)
            m.setdefault(k, []).append(fi)
    return m

def _detect_edges_soft(edge_map, normals, verts,
                       hard_thr_deg: float = 8.0, soft_power: float = 1.5):
    edges, lens, hard, wts = [], [], [], []
    cos_thr = np.cos(np.deg2rad(hard_thr_deg))
    for (i,j), fidxs in edge_map.items():
        v0, v1 = verts[i], verts[j]
        L = np.linalg.norm(v1 - v0)
        if L <= 0:
            continue
        if len(fidxs) < 2:
            edges.append((i,j)); lens.append(L); hard.append(True); wts.append(1.0)
        else:
            f0, f1 = fidxs[0], fidxs[1]
            n0, n1 = normals[f0], normals[f1]
            dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
            ang = np.arccos(dot)                     # [0,π]
            soft = (ang/np.pi)**soft_power          # 0..1
            is_hard = (dot < cos_thr)               # angle > threshold
            edges.append((i,j)); lens.append(L); hard.append(is_hard); wts.append(soft)
    if not edges:
        return (np.empty((0,2),int), np.array([]), np.array([],bool), np.array([]))
    return (np.asarray(edges,int), np.asarray(lens,float),
            np.asarray(hard,bool), np.asarray(wts,float))

def _sample_in_triangle(a,b,c,k):
    if k<=0: return np.empty((0,3))
    r1 = np.random.rand(k); r2 = np.random.rand(k)
    s1 = np.sqrt(r1)
    u = 1 - s1; v = s1*(1-r2); w = s1*r2
    return u[:,None]*a + v[:,None]*b + w[:,None]*c

def _sample_near_edges(a,b,c,k,edge_bias: float = 0.20):
    if k<=0: return np.empty((0,3))
    r1 = np.random.rand(k); r2 = np.random.rand(k)
    s1 = np.sqrt(r1)
    u = 1 - s1; v = s1*(1-r2); w = s1*r2
    bc = np.stack([u,v,w], axis=1)
    idx = np.argmax(bc, axis=1)
    bc[np.arange(k), idx] *= (1.0 - edge_bias)  # pull toward edges
    bc /= np.clip(bc.sum(axis=1, keepdims=True), 1e-12, None)
    u,v,w = bc[:,0], bc[:,1], bc[:,2]
    return u[:,None]*a + v[:,None]*b + w[:,None]*c

def _octant_id(xyz: np.ndarray) -> np.ndarray:
    med = np.median(xyz, axis=0, keepdims=True)
    s = (xyz >= med).astype(int)
    return (s[:,0] << 2) | (s[:,1] << 1) | s[:,2]

# ---------- Orientation helpers (scale-preserving) ----------
def _axis_extents(verts: np.ndarray):
    vmin = verts.min(axis=0); vmax = verts.max(axis=0)
    return (vmax - vmin)

def _auto_orient_car_preserve(verts: np.ndarray) -> np.ndarray:
    """
    Aligns: longest -> Y, second -> X, shortest -> Z.
    Applies rotation about centroid and translates back (so world position preserved).
    """
    cen = verts.mean(axis=0, keepdims=True)
    v = verts - cen
    ext = _axis_extents(v)
    order = np.argsort(-ext)  # descending
    M = np.zeros((3,3))
    M[1, order[0]] = 1.0  # longest -> Y
    M[0, order[1]] = 1.0  # second  -> X
    M[2, order[2]] = 1.0  # shortest-> Z
    if np.linalg.det(M) < 0:
        M[0,:] *= -1.0
    v = v @ M.T
    return v + cen

def _orient_manual_preserve(verts: np.ndarray,
                            axis_permutation: tuple[str,str,str]=('x','y','z'),
                            axis_sign: tuple[int,int,int]=(1,1,1)) -> np.ndarray:
    """
    Permute/flip axes about the centroid (so position is preserved).
    """
    cen = verts.mean(axis=0, keepdims=True)
    v = verts - cen
    idx = {'x':0, 'y':1, 'z':2}
    P = np.eye(3)[:, [idx[a] for a in axis_permutation]]
    S = np.diag(axis_sign)
    M = S @ P
    if np.linalg.det(M) < 0:
        M[0,:] *= -1.0
    v = v @ M.T
    return v + cen

# ---------- Mid-edge subdivision ----------
def subdivide_mesh(verts: np.ndarray, faces: np.ndarray, level: int = 1):
    if level <= 0:
        return verts, faces
    v = verts.copy()
    f = faces.copy()
    for _ in range(level):
        edge_to_mid = {}
        mids = []
        new_faces = []
        def mid_idx(i,j):
            key = (i,j) if i<j else (j,i)
            if key in edge_to_mid:
                return edge_to_mid[key]
            p = 0.5*(v[i] + v[j])
            idx = len(v) + len(mids)
            mids.append(p)
            edge_to_mid[key] = idx
            return idx
        for (i,j,k) in f:
            a = mid_idx(i,j)
            b = mid_idx(j,k)
            c = mid_idx(k,i)
            new_faces.extend([
                [i,a,c],
                [a,j,b],
                [c,b,k],
                [a,b,c]
            ])
        if mids:
            v = np.vstack([v, np.asarray(mids)])
        f = np.asarray(new_faces, dtype=int)
    return v, f

# ---------- Main generator (preserves original scale by default) ----------
def generate_scatterers_from_model(
    model_path: str,
    num_centers: int,
    # orientation (about centroid; preserves position & scale)
    orient: str = "auto",                         # "auto" | "manual" | "none"
    axis_permutation: tuple[str,str,str]=('x','y','z'),
    axis_sign: tuple[int,int,int]=(1,1,1),
    # subdivision
    subdivide_levels: int = 2,
    # surface
    edge_fraction: float = 0.60,
    corner_fraction: float = 0.08,
    min_per_face: int = 3,
    surface_edge_hug_frac: float = 0.50,
    jitter_tangent: float = 0.0012,
    # edges & corners
    hard_edge_threshold_deg: float = 7.0,
    corner_spread: float = 0.006,
    # octant balancing
    octant_floor_surface_frac: float = 0.06,
    octant_floor_edge_frac: float = 0.06,
    # silhouette boost
    silhouette_boost_frac: float = 0.35,
    silhouette_bound_eps: float = 0.06,
    # NEW: scale/centering controls
    preserve_scale: bool = True,                  # keep original dimensions (default)
    center: bool = False,                         # if True, translate final result to origin
    random_seed: Optional[int] = 11
) -> TargetParams:
    """
    View-independent 'polkadot' scatterers with strong shape preservation.
    - Preserves original OBJ dimensions by default (no scaling).
    - Orientation is done about the centroid, then translated back.
    - Set center=True if you want the final points centered at the origin (translation only).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load OBJ
    verts, faces = parse_obj_file(model_path)

    # ---- ORIENT (about centroid, preserves position/scale) ----
    if orient == "auto":
        verts = _auto_orient_car_preserve(verts)
    elif orient == "manual":
        verts = _orient_manual_preserve(verts, axis_permutation, axis_sign)
    # elif "none": keep as-is

    # Optional subdivision (does not change dimensions)
    if subdivide_levels > 0:
        verts, faces = subdivide_mesh(verts, faces, level=subdivide_levels)

    # (No scaling if preserve_scale=True; dimensions remain unchanged)
    # Optional recentering AFTER all geometry ops
    if center:
        verts = verts - verts.mean(axis=0, keepdims=True)

    # Face geometry
    normals, areas, perims = _face_geom(verts, faces)
    if float(areas.sum()) <= 0:
        raise ValueError("Zero-area mesh.")

    # Edge analysis
    e_map = _edge_map(faces)
    e_idx, e_len, e_hard, e_wt = _detect_edges_soft(
        e_map, normals, verts, hard_thr_deg=hard_edge_threshold_deg, soft_power=1.5
    )

    # Corner detection by vertex normal spread
    vert_faces = [[] for _ in range(len(verts))]
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[int(v)].append(fi)
    corner_mask = np.zeros(len(verts), dtype=bool)
    thr_corner = np.deg2rad(max(10.0, hard_edge_threshold_deg + 4.0))
    for vidx, fidxs in enumerate(vert_faces):
        if len(fidxs) < 2:
            corner_mask[vidx] = True
            continue
        fn = normals[fidxs]
        m = fn.mean(axis=0); m /= (np.linalg.norm(m) + 1e-12)
        ang = np.arccos(np.clip(fn @ m, -1, 1))
        if ang.max() > thr_corner:
            corner_mask[vidx] = True
    corners = np.nonzero(corner_mask)[0]

    # Counts
    n_edge   = int(round(num_centers * edge_fraction))
    n_corner = int(round(num_centers * corner_fraction))
    n_surf   = max(0, num_centers - n_edge - n_corner)
    n_corner = min(n_corner, len(corners))

    # ---------- SURFACE: octant-balanced quotas ----------
    face_centroids = np.mean(verts[faces], axis=1)          # (F,3)
    face_oct = _octant_id(face_centroids)
    oct_ids = np.arange(8)
    oct_area = np.array([areas[face_oct==k].sum() for k in oct_ids], float)
    if oct_area.sum() <= 0: oct_area = np.ones(8)/8.0
    oct_probs = oct_area / oct_area.sum()

    oct_floor = max(1, int(octant_floor_surface_frac * max(0, n_surf)))
    oct_quota = np.floor(oct_probs * n_surf).astype(int)
    # floors
    for k in oct_ids:
        need = oct_floor - oct_quota[k]
        if need > 0:
            donors = np.argsort(oct_quota)[::-1]
            for d in donors:
                if d==k or oct_quota[d] <= oct_floor: continue
                give = min(need, oct_quota[d]-oct_floor)
                oct_quota[d]-=give; oct_quota[k]+=give; need-=give
                if need<=0: break
    # exact total
    delta = n_surf - int(oct_quota.sum())
    if delta != 0:
        p = oct_probs / oct_probs.sum()
        idx = np.random.choice(8, size=abs(delta), p=p)
        for k in idx: oct_quota[k] += 1 if delta>0 else -1

    # allocate per-face counts within octants (area-weighted + min_per_face)
    F = len(faces)
    base_counts = np.zeros(F, dtype=int)
    for k in oct_ids:
        mask = (face_oct == k)
        if not np.any(mask): continue
        idxs = np.where(mask)[0]
        quota_k = int(oct_quota[k])
        if quota_k <= 0: continue
        Ak = areas[idxs].sum()
        if Ak <= 0:
            take = np.random.choice(idxs, size=min(quota_k, idxs.size), replace=True)
            for i in take: base_counts[i]+=1
            continue
        wk = areas[idxs] / Ak
        floors = np.full(idxs.size, min_per_face, dtype=int)
        if floors.sum() > quota_k: floors[:] = 0
        base_counts[idxs] += floors
        left = quota_k - floors.sum()
        if left > 0:
            add = np.random.choice(idxs, size=left, p=wk, replace=True)
            for i in add: base_counts[i]+=1

    # ---------- SURFACE: sample (edge-hugging) ----------
    surf_pts = []
    near_frac = float(np.clip(surface_edge_hug_frac, 0.0, 1.0))
    for fi, k in enumerate(base_counts):
        if k <= 0: continue
        a = verts[faces[fi,0]]; b = verts[faces[fi,1]]; c = verts[faces[fi,2]]
        k_edge = int(round(k * near_frac))
        k_int  = k - k_edge
        if k_int  > 0: surf_pts.append(_sample_in_triangle(a,b,c,k_int))
        if k_edge > 0: surf_pts.append(_sample_near_edges(a,b,c,k_edge,edge_bias=0.20))
    surf_pts = np.vstack(surf_pts) if surf_pts else np.empty((0,3))

    # tiny tangent jitter (does not change dimensions meaningfully)
    if jitter_tangent > 0 and surf_pts.shape[0] > 0:
        jit = np.empty_like(surf_pts)
        idx = 0
        # approximate face-based tangent frames
        for fi, k in enumerate(base_counts):
            if k <= 0: continue
            n = normals[fi]
            a = np.array([1.0,0.0,0.0]) if abs(np.dot(n,[1,0,0]))<0.9 else np.array([0.0,1.0,0.0])
            t1 = np.cross(n,a); t1/= (np.linalg.norm(t1)+1e-12)
            t2 = np.cross(n,t1); t2/= (np.linalg.norm(t2)+1e-12)
            offs = (np.random.randn(k,1)*jitter_tangent)*t1 + (np.random.randn(k,1)*jitter_tangent)*t2
            jit[idx:idx+k] = offs; idx += k
        surf_pts = surf_pts + jit

    # ---------- EDGES: octant-balanced + silhouette boost ----------
    edge_pts = np.empty((0,3))
    if e_idx.size>0 and n_edge>0:
        mid = 0.5*(verts[e_idx[:,0]] + verts[e_idx[:,1]])     # (E,3)
        e_oct = _octant_id(mid)

        # silhouette mask: midpoints near bbox min/max along any axis
        vmin = verts.min(axis=0); vmax = verts.max(axis=0); ext = vmax - vmin
        tol = np.maximum(silhouette_bound_eps * np.maximum(ext, 1e-12), 1e-12)
        near_min = (np.abs(mid - vmin) <= tol).any(axis=1)
        near_max = (np.abs(vmax - mid) <= tol).any(axis=1)
        is_sil  = near_min | near_max

        n_sil  = int(round(n_edge * float(np.clip(silhouette_boost_frac,0,1))))
        n_rest = max(0, n_edge - n_sil)

        L = e_len
        hard_w = np.where(e_hard, 1.0, 0.0)
        sel_w_all = L * (0.6*hard_w + 0.4*np.clip(e_wt,0,1))

        # silhouette selection
        if n_sil > 0 and np.any(is_sil):
            idxs = np.where(is_sil)[0]
            w = sel_w_all[idxs]
            w = w / np.clip(w.sum(), 1e-12, None)
            pick = np.random.choice(idxs, size=n_sil, p=w, replace=True)
            t = np.random.rand(pick.size)
            va = verts[e_idx[pick,0]]; vb = verts[e_idx[pick,1]]
            edge_pts_sil = va + t[:,None]*(vb - va)
        else:
            edge_pts_sil = np.empty((0,3))

        # remaining edges: octant-balanced
        edge_pts_rest = np.empty((0,3))
        if n_rest > 0:
            Ltot = np.array([L[e_oct==k].sum() for k in range(8)], float)
            Ltot[Ltot<=0] = 1e-12
            q = np.floor(n_rest * (Ltot/Ltot.sum())).astype(int)

            floor = max(1, int(octant_floor_edge_frac * n_rest))
            for k in range(8):
                need = floor - q[k]
                if need > 0:
                    donors = np.argsort(q)[::-1]
                    for d in donors:
                        if d==k or q[d] <= floor: continue
                        give = min(need, q[d]-floor)
                        q[d]-=give; q[k]+=give; need-=give
                        if need<=0: break

            d = n_rest - q.sum()
            if d != 0:
                p = (Ltot/Ltot.sum()); idx = np.random.choice(8, size=abs(d), p=p)
                for k in idx: q[k] += 1 if d>0 else -1

            sel = []
            for k in range(8):
                mask = (e_oct==k)
                if not np.any(mask) or q[k]<=0: continue
                idxs = np.where(mask)[0]
                w = sel_w_all[idxs]
                w = w / np.clip(w.sum(), 1e-12, None)
                pick = np.random.choice(idxs, size=q[k], p=w, replace=True)
                sel.append(pick)
            if sel:
                sel = np.concatenate(sel)
                t = np.random.rand(sel.size)
                va = verts[e_idx[sel,0]]; vb = verts[e_idx[sel,1]]
                edge_pts_rest = va + t[:,None]*(vb - va)

        edge_pts = np.vstack([edge_pts_sil, edge_pts_rest]) if (edge_pts_sil.size or edge_pts_rest.size) else np.empty((0,3))

    # ---------- CORNERS ----------
    corner_pts = np.empty((0,3))
    if len(corners)>0 and n_corner>0:
        chosen = np.random.choice(corners, size=n_corner, replace=True)
        corner_pts = verts[chosen]
        if corner_spread>0:
            corner_pts = corner_pts + np.random.randn(n_corner,3)*(corner_spread)

    # ---------- Merge & finalize ----------
    pts = np.vstack([surf_pts, edge_pts, corner_pts])
    if pts.shape[0] > num_centers:
        pts = pts[:num_centers]
    elif pts.shape[0] < num_centers:
        need = num_centers - pts.shape[0]
        add = pts[np.random.choice(pts.shape[0], size=need)] + np.random.randn(need,3)*(1e-4)
        pts = np.vstack([pts, add])

    # tiny numerical jitter (won't change dimensions meaningfully)
    pts = pts + (np.random.randn(*pts.shape) * 1e-6)

    amps = np.ones(pts.shape[0])  # “paint dots”; simulator handles physics later

    N = pts.shape[0]
    positions = jnp.array(pts.T)
    return TargetParams(
        positions_m=positions,
        velocities_mps=jnp.zeros((3, N)),
        accelerations_mps2=jnp.zeros((3, N)),
        rcs_dbsm=jnp.array(amps),
        phase_rad=None
    )



target = generate_scatterers_from_model(
    "Cybertruck.obj",
    num_centers=512,
    orient="auto",               # ← auto-detect up = Z, length = Y
    subdivide_levels=2,
    edge_fraction=0.60,
    corner_fraction=0.08,
    min_per_face=3,
    surface_edge_hug_frac=0.55,
    hard_edge_threshold_deg=6.0,
    silhouette_boost_frac=0.45,
    silhouette_bound_eps=0.06,
    jitter_tangent=0.0010,
    octant_floor_surface_frac=0.06,
    octant_floor_edge_frac=0.06,
    random_seed=17
)



plot_scatterers_3d(target)

