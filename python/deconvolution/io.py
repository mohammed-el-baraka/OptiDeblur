"""
Dataset I/O and Management
==========================
Utilities for loading and inspecting benchmark image restoration datasets.
"""

from pathlib import Path
from typing import Dict, Any
import scipy.io as sio
import numpy as np


def load_dataset(dataset_id_or_path: Any) -> Dict[str, np.ndarray]:
    """
    Loads dataset .mat file containing 'Data', 'IR', and 'TrueImage'.

    Parameters
    ----------
    dataset_id_or_path : int or str or Path
        Either 1 or 2 (for Data1.mat, Data2.mat) or a direct file path.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'blurred': 2D ndarray ('Data' in .mat)
        - 'psf': 2D ndarray ('IR' in .mat)
        - 'ground_truth': 2D ndarray ('TrueImage' in .mat)
        - 'name': dataset identifier string
    """
    if str(dataset_id_or_path) in ("1", "Data1", "Data1.mat"):
        path = Path(__file__).resolve().parent.parent.parent / "data" / "Data1.mat"
        name = "Data1"
    elif str(dataset_id_or_path) in ("2", "Data2", "Data2.mat"):
        path = Path(__file__).resolve().parent.parent.parent / "data" / "Data2.mat"
        name = "Data2"
    else:
        path = Path(dataset_id_or_path)
        name = path.stem

    if not path.exists():
        # Try relative to current working directory
        cwd_path = Path("data") / f"{name}.mat"
        if cwd_path.exists():
            path = cwd_path
        else:
            raise FileNotFoundError(f"Dataset file not found at: {path}")

    mat = sio.loadmat(str(path))
    return {
        "blurred": mat["Data"].astype(np.float64),
        "psf": mat["IR"].astype(np.float64),
        "ground_truth": mat["TrueImage"].astype(np.float64),
        "name": name
    }
