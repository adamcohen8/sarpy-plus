import jax.numpy as jnp
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional


def parse_obj_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a simple Wavefront OBJ file into vertices and triangle faces.

    Args:
        file_path: Path to .obj file.

    Returns:
        verts: (V, 3) np.array of vertices [x, y, z].
        faces: (F, 3) np.array of face indices (0-based).

    Assumes only 'v' and 'f' lines, triangle faces, no textures/normals.
    """
    verts = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()[1:]
                verts.append([float(p) for p in parts])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                # Assume 3 verts per face, ignore /tex/norm
                face = [int(p.split('/')[0]) - 1 for p in parts]  # 1-based to 0-based
                if len(face) == 3:
                    faces.append(face)

    if not verts or not faces:
        raise ValueError(f"Invalid OBJ: No verts/faces in {file_path}")

    return np.array(verts), np.array(faces)


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
        edge_bias: float = 0.0  # Future: Fraction of points biased to edges (0-1)
) -> jnp.ndarray:
    """
    Generate scattering centers from a 3D OBJ model.

    Args:
        model_path: Path to .obj file.
        num_centers: Number of scattering centers to sample.
        rcs_scale: Global amplitude/RCS multiplier.
        edge_bias: (Stub) Fraction to place on edges (impl later).

    Returns:
        scatterers: (N, 4) JAX array [x, y, z, amp]
    """
    if edge_bias > 0:
        raise NotImplementedError("Edge bias sampling coming in v0.5")

    verts, faces = parse_obj_file(model_path)
    points = sample_points_on_mesh(verts, faces, num_centers)

    # Assign uniform amplitude for now (edge RCS boosts later)
    amps = np.full(num_centers, rcs_scale)

    scatterers = jnp.hstack([jnp.array(points), jnp.array(amps)[:, jnp.newaxis]])
    return scatterers





def plot_scatterers_3d(
        scatterers: jnp.ndarray,
        cmap: str = 'viridis',
        marker_size: float = 20.0,
        title: str = '3D Scattering Centers',
        save_path: Optional[str] = None
) -> None:
    """
    Plot scattering centers in 3D, colored by amplitude.

    Args:
        scatterers: (N, 4) JAX/NumPy array [x, y, z, amp].
        cmap: Colormap for amplitude (e.g., 'viridis', 'plasma').
        marker_size: Size of scatter points.
        title: Plot title.
        save_path: Optional file path to save the figure (e.g., 'scatter.png').

    Example:
        from sarpy_plus.targets import generate_scatterers_from_model
        scatterers = generate_scatterers_from_model('path/to/model.obj', 500)
        plot_scatterers_3d(scatterers)
    """
    # Convert to NumPy if JAX
    scatterers = jnp.asarray(scatterers)

    if scatterers.shape[1] != 4:
        raise ValueError("Scatterers must be (N, 4) array: [x, y, z, amp]")

    x, y, z, amp = scatterers.T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        x, y, z,
        c=amp,
        cmap=cmap,
        s=marker_size,
        alpha=0.8
    )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Add colorbar for amplitude
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Amplitude / RCS')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()