"""
Utilities for Wannier90 file manipulation, specifically for
bond-centering transformations of the AMN file.

Peter Mlkvik (2024-2025)
"""

import numpy as np
import triqs.utility.mpi as mpi


def apply_bond_transformation(seedname="m1", dirc="./", n_wf=12, n_bands=22, mode="m2"):
    """
    Bond-centering transformation for Wannier90 amn files with modular logic.

    Parameters
    ----------
    seedname : str
        prefix for the Wannier90 files
    rot_logic : callable or np.ndarray
        either a function with signature (n_k, n_wf, kps) returning the rotmat,
        or a pre-computed numpy array of shape (n_k, n_wf, n_wf)
    dirc : str
        directory path
    n_wf : int
        number of Wannier functions
    n_bands : int
        number of bands
    """

    # ensure file IO only happens on master node to prevent race conditions
    if not mpi.is_master_node():
        return

    # load k-points from .win file
    win_path = f"{dirc}{seedname}.win"
    with open(win_path) as file:
        content = file.read()
        kps_txt = (
            content.split("begin kpoints")[1]
            .split("end kpoints")[0]
            .strip()
            .split("\n")
        )
        kps = [[float(k) for k in line.split()] for line in kps_txt]

    n_k = len(kps)
    amn_path = f"{dirc}{seedname}.amn"

    # load amn data
    amn_head = np.genfromtxt(amn_path, dtype="str", max_rows=1)
    if amn_head[2] == "0":
        mpi.report(f"Warning: {seedname}.amn already has the processed flag.")

    amn_data = np.loadtxt(amn_path, skiprows=2)
    save_shape = amn_data.shape
    assert amn_data.shape[0] == n_k * n_bands * n_wf, "Dimension mismatch in amn file!"

    # reshape and extract complex projections
    amn_data = amn_data.reshape((n_k, n_wf, n_bands, 5))
    projmat_amn = amn_data[:, :, :, 3] + 1j * amn_data[:, :, :, 4]

    # build Rotation Matrix based on mode
    rotmat = np.zeros((n_k, n_wf, n_wf), dtype=complex)

    if mode == "m2":
        rotmat = _build_rotmat_m2(n_k, n_wf, kps)
    elif mode == "m1":
        rotmat = _build_rotmat_m1(n_k, n_wf, kps)
    else:
        raise ValueError("Unknown mode. Use 'm1' or 'm2'.")

    # apply transformation
    # A' = sqrt(0.5) * U * A
    transformed = np.sqrt(0.5) * np.einsum("kab,kbc->kac", rotmat, projmat_amn)

    # prepare and save
    amn_data[:, :, :, 3] = np.real(transformed)
    amn_data[:, :, :, 4] = np.imag(transformed)

    header = f" Bond-centered \n{n_bands:10d}{n_k:10d}{n_wf:10d}"

    np.savetxt(
        amn_path,
        amn_data.reshape(save_shape),
        fmt=["%d", "%d", "%d", "%1.12f", "%1.12f"],
        header=header,
        comments="",
    )

    mpi.report(f"  solid_dmft: Successfully transformed {seedname}.amn")


def _build_rotmat_m2(n_k, n_wf, kps):
    rotmat = np.array([np.eye(n_wf, dtype=complex)] * n_k)
    dimer_dir = [0, 1, 0]
    starting_phase = 1j * np.pi / 4
    delta = -1 / 4

    for i in range(n_k):
        phase = 1j * 2 * np.pi * np.dot(kps[i], dimer_dir)
        phases = {
            "00": delta * phase,
            "03": -delta * phase,
            "30": (delta - 0.5) * phase,
            "33": (-delta - 0.5) * phase,
        }

        # Orbital indices and offsets
        for orb in range(3):
            for b in [3, 6, 12, 18]:  # bottom
                rotmat[i, b + orb, b + orb] = np.exp(starting_phase + phases["00"])
            for t in [0, 9, 15, 21]:  # top
                rotmat[i, t + orb, t + orb] = np.exp(starting_phase + phases["33"])

            # Off-diagonal couplings
            rotmat[i, 0 + orb, 6 + orb] = np.exp(-starting_phase + phases["30"])
            rotmat[i, 3 + orb, 9 + orb] = np.exp(-starting_phase + phases["03"])
            rotmat[i, 6 + orb, 0 + orb] = np.exp(-starting_phase + phases["03"])
            rotmat[i, 9 + orb, 3 + orb] = np.exp(-starting_phase + phases["30"])
            # ... (repeat for 12, 21, 15, 18 as per your original code)
            rotmat[i, 12 + orb, 21 + orb] = np.exp(-starting_phase + phases["03"])
            rotmat[i, 15 + orb, 18 + orb] = np.exp(-starting_phase + phases["30"])
            rotmat[i, 18 + orb, 15 + orb] = np.exp(-starting_phase + phases["03"])
            rotmat[i, 21 + orb, 12 + orb] = np.exp(-starting_phase + phases["30"])
    return rotmat


def _build_rotmat_m1(n_k, n_wf, kps):
    rotmat = np.zeros((n_k, n_wf, n_wf), dtype=complex)
    dimer_dir = [1, 0, 0]
    for i in range(n_k):
        dot_top = np.dot(kps[i], dimer_dir)
        p_top = np.exp(1j * 2 * np.pi * dot_top)
        p_bot = 1.0  # 0*dot results in exp(0)

        # Sub-block assignments
        for offset, p in [(0, p_top), (6, p_top)]:
            rotmat[i, 0 + offset, 3 + offset] = p
            rotmat[i, 1 + offset, 4 + offset] = -p
            rotmat[i, 2 + offset, 5 + offset] = p

        for offset, p in [(0, p_bot), (6, p_bot)]:
            rotmat[i, 3 + offset, 0 + offset] = p
            rotmat[i, 4 + offset, 1 + offset] = -p
            rotmat[i, 5 + offset, 2 + offset] = p
    return rotmat
