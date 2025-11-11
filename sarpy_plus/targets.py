import jax.numpy as jnp
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
from sarpy_plus.params import TargetParams


def parse_obj_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a simple Wavefront OBJ file into vertices and triangle faces.

    Args:
        file_path: Path to .obj file.

    Returns:
        verts: (V, 3) np.array of vertices [x, y, z].
        faces: (F, 3) np.array of face indices (0-based).

    Assumes only 'v' and 'f' lines, auto-triangulates n-gons, no textures/normals.
    """
    verts = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()[1:]
                if len(parts) < 3:
                    raise ValueError(f"Invalid vertex in {file_path}: {line}")
                # Take only x y z, ignore r g b or w if present
                verts.append([float(p) for p in parts[:3]])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                # Parse indices, ignore /tex/norm
                face = [int(p.split('/')[0]) - 1 for p in parts]  # 1-based to 0-based
                if len(face) < 3:
                    raise ValueError(f"Invalid face (<3 verts) in {file_path}: {line}")
                # Fan-triangulate if n-gon
                for i in range(1, len(face) - 1):
                    faces.append([face[0], face[i], face[i + 1]])

    verts = np.array(verts)
    faces = np.array(faces)

    if verts.shape[1] != 3:
        raise ValueError(f"Verts not 3D: shape {verts.shape}")

    if len(faces) == 0:
        raise ValueError(f"No faces found in {file_path}")

    return verts, faces


def compute_triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute areas of triangles for weighted sampling.
    """
    ab = verts[faces[:, 1]] - verts[faces[:, 0]]
    ac = verts[faces[:, 2]] - verts[faces[:, 0]]
    cross = np.cross(ab, ac)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    return areas


def sample_points_on_mesh(verts: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """
    Sample points uniformly on mesh surface, weighted by triangle area.

    Returns: (N, 3) np.array [x, y, z]
    """
    areas = compute_triangle_areas(verts, faces)
    if np.sum(areas) == 0:
        raise ValueError("Mesh has zero area triangles")

    # Probabilities for choosing triangles
    probs = areas / np.sum(areas)

    # Choose triangles
    chosen_faces = np.random.choice(len(faces), size=num_points, p=probs)

    # Barycentric coords: uniform random on triangle
    r1 = np.random.rand(num_points)
    r2 = np.random.rand(num_points)
    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2

    # Interpolate points
    va = verts[faces[chosen_faces, 0]]
    vb = verts[faces[chosen_faces, 1]]
    vc = verts[faces[chosen_faces, 2]]
    points = u[:, np.newaxis] * va + v[:, np.newaxis] * vb + w[:, np.newaxis] * vc

    return points


def generate_scatterers_from_model(
        model_path: str,
        num_centers: int,
        rcs_scale: float = 1.0,
        edge_bias: float = 0.0,
        edge_rcs_boost: float = 1.0  # Multiplier for edge point amps
) -> TargetParams:
    """
    Generate scattering centers from a 3D OBJ model in TargetParams format.

    Args:
        model_path: Path to .obj file.
        num_centers: Number of scattering centers to sample.
        rcs_scale: Global amplitude/RCS multiplier (in dBsm).
        edge_bias: Fraction of points to sample on edges (0-1).
        edge_rcs_boost: Amplitude boost for edge-sampled points.

    Returns:
        TargetParams instance with positions (3, N), velocities/accelerations as zeros (3, N), rcs_dbsm (N,).
    """
    if not 0 <= edge_bias <= 1:
        raise ValueError("edge_bias must be between 0 and 1")

    verts, faces = parse_obj_file(model_path)

    num_edge = int(num_centers * edge_bias)
    num_surf = num_centers - num_edge

    # Surface points (if any)
    surf_points = np.empty((0, 3)) if num_surf == 0 else sample_points_on_mesh(verts, faces, num_surf)
    surf_amps = np.full(len(surf_points), rcs_scale)

    # Edge points (if any)
    edge_points = np.empty((0, 3))
    if num_edge > 0:
        # Get unique edges (undirected)
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.sort(edges, axis=1)  # Sort for uniqueness
        edges = np.unique(edges, axis=0)  # (E, 2) unique edges

        # Edge lengths for weighting
        va = verts[edges[:, 0]]
        vb = verts[edges[:, 1]]
        lengths = np.linalg.norm(vb - va, axis=1)
        if np.sum(lengths) == 0:
            raise ValueError("Mesh has zero-length edges")
        probs = lengths / np.sum(lengths)

        # Choose edges
        chosen_edges = np.random.choice(len(edges), size=num_edge, p=probs)

        # Sample points along chosen edges
        t = np.random.rand(num_edge)  # [0,1] lerp factor
        va = verts[edges[chosen_edges, 0]]
        vb = verts[edges[chosen_edges, 1]]
        edge_points = va + t[:, np.newaxis] * (vb - va)

    edge_amps = np.full(len(edge_points), rcs_scale * edge_rcs_boost)

    # Combine
    points = np.vstack([surf_points, edge_points])
    amps = np.hstack([surf_amps, edge_amps])

    if len(points) != num_centers:
        raise ValueError(f"Generated {len(points)} points, expected {num_centers}")

    # Convert to TargetParams format
    N = num_centers
    positions = jnp.array(points.T)  # (3, N)
    velocities = jnp.zeros((3, N))
    accelerations = jnp.zeros((3, N))
    rcs_dbsm = jnp.array(amps)  # (N,)

    return TargetParams(
        positions_m=positions,
        velocities_mps=velocities,
        accelerations_mps2=accelerations,
        rcs_dbsm=rcs_dbsm,
        phase_rad=None
    )


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
    ax.view_init(elev=20, azim=-60)
    ax.set_title(title)

    # Add colorbar for amplitude
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Amplitude / RCS (dBsm)')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()