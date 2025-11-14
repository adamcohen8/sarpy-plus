# ------------------------------------------------------------
# scatterer_generator.py
# Returns: TargetParams, BVH, ScattererMeta
# - Preserves original OBJ dimensions
# - Orientation about centroid (auto/manual/none)
# - Optional mid-edge subdivision (shape-preserving)
# - Surface (edge-hugging), edge, and corner sampling
# - Octant balancing + silhouette boost (view-independent)
# - Records per-dot metadata for RCS computation
# ------------------------------------------------------------

from typing import Tuple, Optional, List
import numpy as np
import jax.numpy as jnp
from sarpy_plus.params import TargetParams
from sarpy_plus.sim import BVH              # <-- adjust import if BVH lives elsewhere
from sarpy_plus.params import ScattererMeta    # <-- from the Level-1 RCS module we set up
import matplotlib.pyplot as plt


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
    """Map edge (i,j) -> list of adjacent face indices"""
    m = {}
    for fi, f in enumerate(faces):
        e = [(int(f[0]),int(f[1])), (int(f[1]),int(f[2])), (int(f[2]),int(f[0]))]
        for i,j in e:
            k = (i,j) if i<j else (j,i)
            m.setdefault(k, []).append(fi)
    return m

def _detect_edges_soft(edge_map, normals, verts,
                       hard_thr_deg: float = 8.0, soft_power: float = 1.5):
    edges, lens, hard, wts, dihedral = [], [], [], [], []
    cos_thr = np.cos(np.deg2rad(hard_thr_deg))
    for (i,j), fidxs in edge_map.items():
        v0, v1 = verts[i], verts[j]
        L = np.linalg.norm(v1 - v0)
        if L <= 0:
            continue
        if len(fidxs) < 2:
            # boundary → treat as hard with π dihedral
            edges.append((i,j)); lens.append(L); hard.append(True); wts.append(1.0); dihedral.append(np.pi)
        else:
            f0, f1 = fidxs[0], fidxs[1]
            n0, n1 = normals[f0], normals[f1]
            dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
            ang = np.arccos(dot)                      # 0..π (dihedral supplement)
            soft = (ang/np.pi)**soft_power
            is_hard = (dot < cos_thr)
            edges.append((i,j)); lens.append(L); hard.append(is_hard); wts.append(soft); dihedral.append(ang)
    if not edges:
        return (np.empty((0,2),int), np.array([]), np.array([],bool), np.array([]), np.array([]))
    return (np.asarray(edges,int), np.asarray(lens,float),
            np.asarray(hard,bool), np.asarray(wts,float), np.asarray(dihedral,float))

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


# ---------- Orientation (preserve scale & placement) ----------
def _axis_extents(verts: np.ndarray):
    vmin = verts.min(axis=0); vmax = verts.max(axis=0)
    return (vmax - vmin)

def _auto_orient_car_preserve(verts: np.ndarray) -> np.ndarray:
    cen = verts.mean(axis=0, keepdims=True)
    v = verts - cen
    ext = _axis_extents(v)
    order = np.argsort(-ext)  # longest, second, shortest
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


# ---------- Simple estimates for corner metadata ----------
def estimate_corner_size(v_idx: int, verts: np.ndarray, edge_map) -> float:
    Ls = []
    for (i,j), _ in edge_map.items():
        if i==v_idx or j==v_idx:
            Ls.append(np.linalg.norm(verts[i]-verts[j]))
    return float(np.mean(Ls)) if Ls else 0.0

def estimate_bisector_from_faces(face_ids: List[int], normals: np.ndarray) -> np.ndarray:
    if not face_ids:
        return np.zeros(3)
    fn = normals[np.array(face_ids)]
    b = fn.mean(axis=0)
    n = np.linalg.norm(b)
    return (b / n) if n > 0 else np.zeros(3)


# ---------- Main generator (returns TargetParams, BVH, ScattererMeta) ----------
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
    # placement
    center: bool = False,                         # translate final result to origin (no scaling ever)
    # BVH control
    bvh_leaf_size: int = 8,
    # RNG
    random_seed: Optional[int] = 11
) -> Tuple[TargetParams, BVH, ScattererMeta]:
    """
    Returns:
      target: TargetParams (positions (3,N), zeros v/a, unit amps)
      bvh:    BVH built from the oriented, subdivided mesh (same frame as positions)
      meta:   ScattererMeta with per-dot fields (facet/edge/corner)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # --- Load & orient (preserve dimensions) ---
    verts, faces = parse_obj_file(model_path)
    if orient == "auto":
        verts = _auto_orient_car_preserve(verts)
    elif orient == "manual":
        verts = _orient_manual_preserve(verts, axis_permutation, axis_sign)
    # elif "none": keep as-is

    # Subdivide (shape-preserving)
    if subdivide_levels > 0:
        verts, faces = subdivide_mesh(verts, faces, level=subdivide_levels)

    # Optional recenter AFTER orientation/subdivision
    if center:
        verts = verts - verts.mean(axis=0, keepdims=True)

    # Build BVH on this exact mesh (same frame as scatterers)
    bvh = BVH.from_mesh(verts, faces, leaf_size=bvh_leaf_size)

    # Face geometry
    normals, areas, perims = _face_geom(verts, faces)
    if float(areas.sum()) <= 0:
        raise ValueError("Zero-area mesh.")

    # Edge analysis (+ dihedral angle)
    e_map = _edge_map(faces)
    e_idx, e_len, e_hard, e_wt, e_dih = _detect_edges_soft(
        e_map, normals, verts, hard_thr_deg=hard_edge_threshold_deg, soft_power=1.5
    )

    # Corner detection (vertex normal spread) + per-vertex face lists
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
    # floors & exact total
    for k in oct_ids:
        need = oct_floor - oct_quota[k]
        if need > 0:
            donors = np.argsort(oct_quota)[::-1]
            for d in donors:
                if d==k or oct_quota[d] <= oct_floor: continue
                give = min(need, oct_quota[d]-oct_floor)
                oct_quota[d]-=give; oct_quota[k]+=give; need-=give
                if need<=0: break
    delta = n_surf - int(oct_quota.sum())
    if delta != 0:
        p = (oct_probs / oct_probs.sum())
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

    # ---------- Prepare metadata accumulators ----------
    kinds: List[int] = []
    facet_normals: List[np.ndarray] = []
    facet_area_shares: List[float] = []
    facet_rough: List[float] = []

    edge_len_shares: List[float] = []
    wedge_alphas: List[float] = []

    corner_sizes_a: List[float] = []
    corner_bisectors: List[np.ndarray] = []

    # ---------- SURFACE: sample (edge-hugging) + metadata ----------
    surf_pts = []
    near_frac = float(np.clip(surface_edge_hug_frac, 0.0, 1.0))
    for fi, k in enumerate(base_counts):
        if k <= 0: continue
        a = verts[faces[fi,0]]; b = verts[faces[fi,1]]; c = verts[faces[fi,2]]
        k_edge = int(round(k * near_frac))
        k_int  = k - k_edge

        pts_face = []
        if k_int  > 0: pts_face.append(_sample_in_triangle(a,b,c,k_int))
        if k_edge > 0: pts_face.append(_sample_near_edges(a,b,c,k_edge,edge_bias=0.20))
        if pts_face:
            P = np.vstack(pts_face)
            surf_pts.append(P)

            # Metadata for facets:
            # assign area share = face_area / k (for that face)
            area_share = float(areas[fi]) / max(k, 1)
            n = normals[fi]
            # append k entries
            kinds.extend([0] * k)
            facet_normals.extend([n] * k)
            facet_area_shares.extend([area_share] * k)
            facet_rough.extend([0.0] * k)  # tweak if you have per-face roughness

            # pad non-used fields
            edge_len_shares.extend([0.0] * k)
            wedge_alphas.extend([0.0] * k)
            corner_sizes_a.extend([0.0] * k)
            corner_bisectors.extend([np.zeros(3)] * k)

    surf_pts = np.vstack(surf_pts) if surf_pts else np.empty((0,3))

    # tiny tangent jitter (does not change dimensions meaningfully)
    if jitter_tangent > 0 and surf_pts.shape[0] > 0:
        # per-face jitter already approximated in earlier version; keep a global tiny noise
        surf_pts = surf_pts + (np.random.randn(*surf_pts.shape) * jitter_tangent)

    # ---------- EDGES: silhouette boost + octant balancing + metadata ----------
    edge_pts = np.empty((0,3))
    if e_idx.size>0 and n_edge>0:
        mid = 0.5*(verts[e_idx[:,0]] + verts[e_idx[:,1]])     # (E,3)
        e_oct = _octant_id(mid)

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

        chosen_edges_total: List[int] = []
        chosen_pts_total: List[np.ndarray] = []

        # silhouette selection
        if n_sil > 0 and np.any(is_sil):
            idxs = np.where(is_sil)[0]
            w = sel_w_all[idxs]
            w = w / np.clip(w.sum(), 1e-12, None)
            pick = np.random.choice(idxs, size=n_sil, p=w, replace=True)
            t = np.random.rand(pick.size)
            va = verts[e_idx[pick,0]]; vb = verts[e_idx[pick,1]]
            pts_sil = va + t[:,None]*(vb - va)
            chosen_edges_total.extend(pick.tolist())
            chosen_pts_total.append(pts_sil)

        # remaining edges: octant-balanced
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

            for k in range(8):
                mask = (e_oct==k)
                if not np.any(mask) or q[k]<=0: continue
                idxs = np.where(mask)[0]
                w = sel_w_all[idxs]
                w = w / np.clip(w.sum(), 1e-12, None)
                pick = np.random.choice(idxs, size=q[k], p=w, replace=True)
                t = np.random.rand(pick.size)
                va = verts[e_idx[pick,0]]; vb = verts[e_idx[pick,1]]
                pts_rest = va + t[:,None]*(vb - va)
                chosen_edges_total.extend(pick.tolist())
                chosen_pts_total.append(pts_rest)

        if chosen_pts_total:
            edge_pts = np.vstack(chosen_pts_total)
            # ---- Edge metadata assignment ----
            chosen_edges_total = np.asarray(chosen_edges_total, dtype=int)
            # count how many samples landed on each chosen edge so we can assign length share
            uniq, cnts = np.unique(chosen_edges_total, return_counts=True)
            cnt_map = {int(u): int(c) for u,c in zip(uniq, cnts)}
            # for each selected edge occurrence, append meta:
            for e_id in chosen_edges_total:
                L_share = float(L[e_id]) / max(cnt_map[e_id], 1)
                kinds.append(1)
                # facet fields not used
                facet_normals.append(np.zeros(3))
                facet_area_shares.append(0.0)
                facet_rough.append(0.0)
                # edge fields
                edge_len_shares.append(L_share)
                wedge_alphas.append(float(e_dih[e_id]))
                # corner fields not used
                corner_sizes_a.append(0.0)
                corner_bisectors.append(np.zeros(3))

    # ---------- CORNERS + metadata ----------
    corner_pts = np.empty((0,3))
    if len(corners)>0 and n_corner>0:
        chosen = np.random.choice(corners, size=n_corner, replace=True)
        corner_pts = verts[chosen]
        if corner_spread>0:
            corner_pts = corner_pts + np.random.randn(n_corner,3)*(corner_spread)

        # metadata for corners
        for v_idx in chosen:
            a_size = estimate_corner_size(int(v_idx), verts, e_map)
            b_vec  = estimate_bisector_from_faces(vert_faces[int(v_idx)], normals)
            kinds.append(2)
            # facet fields not used
            facet_normals.append(np.zeros(3))
            facet_area_shares.append(0.0)
            facet_rough.append(0.0)
            # edge fields not used
            edge_len_shares.append(0.0)
            wedge_alphas.append(0.0)
            # corner fields
            corner_sizes_a.append(float(a_size))
            corner_bisectors.append(b_vec)

    # ---------- Merge & finalize ----------
    pts = np.vstack([surf_pts, edge_pts, corner_pts])
    if pts.shape[0] > num_centers:
        pts = pts[:num_centers]
        # trim metadata to match
        keep = pts.shape[0]
        kinds            = kinds[:keep]
        facet_normals    = facet_normals[:keep]
        facet_area_shares= facet_area_shares[:keep]
        facet_rough      = facet_rough[:keep]
        edge_len_shares  = edge_len_shares[:keep]
        wedge_alphas     = wedge_alphas[:keep]
        corner_sizes_a   = corner_sizes_a[:keep]
        corner_bisectors = corner_bisectors[:keep]
    elif pts.shape[0] < num_centers and pts.shape[0] > 0:
        need = num_centers - pts.shape[0]
        idx = np.random.choice(pts.shape[0], size=need)
        pts = np.vstack([pts, pts[idx] + np.random.randn(need,3)*(1e-4)])
        # duplicate corresponding metadata
        kinds           += [kinds[i] for i in idx]
        facet_normals   += [facet_normals[i] for i in idx]
        facet_area_shares += [facet_area_shares[i] for i in idx]
        facet_rough     += [facet_rough[i] for i in idx]
        edge_len_shares += [edge_len_shares[i] for i in idx]
        wedge_alphas    += [wedge_alphas[i] for i in idx]
        corner_sizes_a  += [corner_sizes_a[i] for i in idx]
        corner_bisectors+= [corner_bisectors[i] for i in idx]

    # tiny numerical jitter
    if pts.shape[0] > 0:
        pts = pts + (np.random.randn(*pts.shape) * 1e-6)

    amps = np.ones(pts.shape[0])  # simulator handles physics later

    N = pts.shape[0]
    positions = jnp.array(pts.T)

    target = TargetParams(
        positions_m=positions,
        velocities_mps=jnp.zeros((3, N)),
        accelerations_mps2=jnp.zeros((3, N)),
        rcs_dbsm=jnp.array(amps),
        phase_rad=None
    )

    # ---- Pack metadata to JAX arrays ----
    meta = ScattererMeta(
        kind=jnp.asarray(np.array(kinds), jnp.int32),
        face_normal=jnp.asarray(np.array(facet_normals), jnp.float32),
        area_share=jnp.asarray(np.array(facet_area_shares), jnp.float32),
        roughness=jnp.asarray(np.array(facet_rough), jnp.float32),
        edge_len_share=jnp.asarray(np.array(edge_len_shares), jnp.float32),
        wedge_alpha=jnp.asarray(np.array(wedge_alphas), jnp.float32),
        corner_size_a=jnp.asarray(np.array(corner_sizes_a), jnp.float32),
        corner_bisector=jnp.asarray(np.array(corner_bisectors), jnp.float32),
    )

    return target, bvh, meta




def plot_scatterers_3d(
        target: TargetParams,
        cmap: str = 'viridis',
        marker_size: float = 20.0,
        title: str = '3D Scattering Centers',
        save_path: Optional[str] = None
) -> None:
    """
    Plot scattering centers in 3D, colored by amplitude.

    Args:
        target: TargetParams instance with positions (3, N) and rcs_dbsm (N,).
        cmap: Colormap for amplitude (e.g., 'viridis', 'plasma').
        marker_size: Size of scatter points.
        title: Plot title.
        save_path: Optional file path to save the figure (e.g., 'scatter.png').

    Example:
        from sarpy_plus.targets import generate_scatterers_from_model
        target = generate_scatterers_from_model('path/to/model.obj', 500)
        plot_scatterers_3d(target)
    """
    # Extract positions and amps
    positions = jnp.asarray(target.positions_m.T)  # (N, 3)
    amps = jnp.asarray(target.rcs_dbsm)  # (N,)

    if positions.shape[1] != 3:
        raise ValueError("Positions must be (N, 3)")
    if len(amps) != len(positions):
        raise ValueError("Amps length must match number of positions")

    x, y, z = positions.T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        x, y, z,
        c=amps,
        cmap=cmap,
        s=marker_size,
        alpha=0.8
    )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=0, azim=0)
    ax.set_title(title)

    # Add colorbar for amplitude
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Amplitude / RCS (dBsm)')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()