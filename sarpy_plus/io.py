from typing import Optional, Tuple
import jax.numpy as jnp
import numpy as np

SARPY_AVAILABLE = False
try:
    from sarpy.io.complex.converter import open_complex

    SARPY_AVAILABLE = True
except ImportError:
    pass  # We'll raise error only on use


def load_from_sicd(file_path: str, index: int = 0) -> Tuple[jnp.ndarray, Optional[dict]]:
    """
    Load complex SAR data and metadata from a SICD (or similar) file using NGA Sarpy.

    Args:
        file_path: Path to the SICD file (e.g., .nitf).
        index: Image index for multi-image files (default 0).

    Returns:
        data: JAX array of complex pixel data.
        meta: Dict of SICD metadata (or None if no meta).

    Raises:
        ImportError: If Sarpy is not installed.
        ValueError: If file can't be opened or read.
    """
    if not SARPY_AVAILABLE:
        raise ImportError(
            "NGA Sarpy library not installed. Install with `pip install sarpy-plus[sarpy]` "
            "or `pip install sarpy`."
        )

    try:
        reader = open_complex(file_path)
    except Exception as e:
        raise ValueError(f"Failed to open {file_path} with Sarpy: {e}")

    # Read complex data (numpy complex64) and convert to JAX
    data_np = reader[:, :, index]  # Full data; use slicing for chips, e.g., reader[:500, :500, index]
    data = jnp.array(data_np)

    # Get metadata as dict (from first SICD structure if multi)
    sicd_tuple = reader.get_sicds_as_tuple()
    meta = sicd_tuple[index].to_dict() if sicd_tuple else None

    return data, meta