from typing import Optional, Union

import cc3d
import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from TPTBox import NII


def mincutmaxflow(
    vertebra_nii: NII,
    separator_ivd: NII,
    connectivity: int = 1,
) -> NII:
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"
    connectivity = min(connectivity * 2, 8) if vertebra_nii.ndim == 2 else 6 if connectivity == 1 else 18 if connectivity == 2 else 26
    vol = vertebra_nii.get_seg_array()
    return vertebra_nii.set_array(
        split_cc(
            vol=vol,
            sep=separator_ivd.get_seg_array() if separator_ivd is not None else None,
            connectivity=connectivity,
            structure=generate_binary_structure(vol.ndim, connectivity),
            min_vol=10,
            voxel_dim=np.asarray(vertebra_nii.zoom),
        )
    )


def np_mincutmaxflow(
    vertebra_arr: np.ndarray,
    separator_ivd_arr: np.ndarray | None,
    connectivity: int = 1,
    zoom: np.ndarray | None = None,
) -> np.ndarray:
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"
    connectivity = min(connectivity * 2, 8) if vertebra_arr.ndim == 2 else 6 if connectivity == 1 else 18 if connectivity == 2 else 26
    return split_cc(
        vol=vertebra_arr,
        sep=separator_ivd_arr if separator_ivd_arr is not None else None,
        connectivity=connectivity,
        structure=generate_binary_structure(vertebra_arr.ndim, connectivity),
        min_vol=10,
        voxel_dim=np.asarray(zoom) if zoom is not None else None,
    )


def split_cc(  # noqa: C901
    vol: np.ndarray,
    sep: np.ndarray | None = None,
    connectivity: int = 6,
    structure: None | np.ndarray | list[np.ndarray] = None,
    min_vol: int | None = None,
    max_cut: int | None = None,
    max_ignore: int | None = 6,
    voxel_dim: np.ndarray | None = None,
    add_2d_edges: bool = True,
) -> np.ndarray:
    """

    @param vol: volume which only contains values of one connected component
    @param sep: if given, vol is will not be eroded, sep will be dilated.
    @param connectivity: 6 (voxel faces), 18 (+edges), or 26 (+corners)
    @param structure: 3d numpy array of type true, with which the volume should be eroded
    @param min_vol: minimal size of the both eroded
    @param max_cut:
    @param voxel_dim: weights along the dimensions (x y and z) for cost function (if not set, all are 1.0)
    @param add_2d_edges: if True, not only add edges to left/up/depth,.. also to left+up,up+depth,left+depth,..
    @return:
    """
    _, m = cc3d.connected_components(vol, connectivity=connectivity, return_N=True)
    if m != 1:
        raise Exception(f"volume is separable into {m} parts with {connectivity=} - it should be 1.")  # noqa: TRY002
    vol_erode = vol
    iterations = 0
    # if sep is not None:
    #     sep = np.invert(sep)
    if isinstance(structure, np.ndarray):
        structure = [structure]
    while True:
        # vol_erode_old = vol_erode
        structure_acctual = structure[iterations % len(structure)] if structure is not None else None
        if sep is not None:
            sep = binary_dilation(sep, structure=structure_acctual)
            # vol_erode = vol & sep
            vol_erode = np.where(sep, 0, vol)
        else:
            vol_erode = binary_erosion(vol_erode, structure=structure_acctual)
        cc_erode, m = cc3d.connected_components(vol_erode, connectivity=connectivity, return_N=True)
        iterations += 1
        if m > 1 and max_ignore is not None:
            res = np.unique(cc_erode, return_counts=True)
            max_errors = sum([s for x, s in zip(*res, strict=True) if x > 0 and s <= max_ignore])
            idxs = [x for x, s in zip(*res, strict=True) if x > 0 and s > max_ignore]
            m = len(idxs)
            if m > 2:
                break  # should result in error "erosion with struc.."
            if m == 2:
                cc_erode_ = np.zeros(cc_erode.shape, dtype=cc_erode.dtype)
                cc_erode_[cc_erode == idxs[0]] = 1
                cc_erode_[cc_erode == idxs[1]] = 2
                cc_erode = cc_erode_
                break
            # otherwise, contiue!
        if m == 0:
            raise Exception(  # noqa: TRY002
                f"cannot split volume into two parts after {iterations} iterations, all values are 0 after erosion."
            )
    if m > 2:
        raise Exception(  # noqa: TRY002
            f"erosion with struture {structure} leads to {m} separate connected components after {iterations} isterations, expect 2."
        )

    S = cc_erode == 1  # noqa: N806
    T = cc_erode == 2  # noqa: N806
    G_ = vol ^ vol_erode  # noqa: N806

    if min_vol is not None and S.sum() < min_vol:
        raise Exception(  # noqa: TRY002
            f"after erosion for split, volume of one structure is {S.sum()} which is smaller than the accepted size {min_vol}."
        )
    if min_vol is not None and T.sum() < min_vol:
        raise Exception(  # noqa: TRY002
            f"after erosion for split, volume of one structure is {T.sum()} which is smaller than the accepted size {min_vol}."
        )
    if voxel_dim is None:
        voxel_dim = np.ones([3])
        capacity_end = 1000
    else:
        # this is max(x*y,y*z,x*z)*1000
        capacity_end = np.prod(voxel_dim) / np.min(voxel_dim) * 1000

    S_dil = binary_dilation(S, structure=structure_acctual)  # noqa: N806
    T_dil = binary_dilation(T, structure=structure_acctual)  # noqa: N806
    to_S = np.argwhere(S_dil & G_)  # noqa: N806
    to_T = np.argwhere(T_dil & G_)  # noqa: N806
    if len(to_T) == 0 or len(to_S) == 0:
        raise Exception("no connection between separated objects and remaining vertices found")  # noqa: TRY002

    G = nx.Graph()  # noqa: N806
    import itertools

    for x, y, z in itertools.product([0, 1], repeat=3):
        vec = np.array([x, y, z])
        xe, ye, ze = np.array(G_.shape) - vec

        def add_edges(points, diff1, diff2, cap):
            # print(f"Add edge {points}, {diff1}, {diff2}, {cap}")
            """Add edge from point + diff1 to point + diff2 for each point in points. Must be tuples because
            numpy arrays are note hashable, and nodes in the graph have to be hashable."""
            G.add_edges_from([*zip(map(tuple, points + diff1), map(tuple, points + diff2), strict=False)], capacity=cap)

        if x + y + z == 1:
            # calculate the product of the both dimension, which are
            # for x=1, y=z=0 it is y*z
            capacity = np.prod(voxel_dim[vec == 0])
            add_edges(np.argwhere(G_[x:, y:, z:] & G_[:xe, :ye, :ze]), [0, 0, 0], [x, y, z], capacity)
        if x + y + z == 2 and add_2d_edges:
            # calculate a plane diagonal in two dimension and direct into the other dimension
            # for x=y=1, z=0 it is sqrt(x^2+y^2)*z
            capacity = voxel_dim[vec == 0] * np.linalg.norm(voxel_dim[vec == 1])
            add_edges(np.argwhere(G_[x:, y:, z:] & G_[:xe, :ye, :ze]), [0, 0, 0], [x, y, z], capacity)
            if x == 1:
                add_edges(np.argwhere(G_[:xe, y:, z:] & G_[x:, :ye, :ze]), [x, 0, 0], [0, y, z], capacity)
            else:
                add_edges(np.argwhere(G_[x:, :ye, z:] & G_[:xe, y:, :ze]), [0, y, 0], [x, 0, z], capacity)

    G.add_edges_from([((x, y, z), "t") for x, y, z in to_T], capacity=capacity_end)
    G.add_edges_from([("s", (x, y, z)) for x, y, z in to_S], capacity=capacity_end)

    if not nx.has_path(G, "s", "t"):
        raise Exception("no path exists from s to t")  # noqa: TRY002

    cut_value, (s_idx, t_idx) = nx.minimum_cut(G, "s", "t")
    if max_cut is not None and cut_value > max_cut:
        raise Exception(f"cut size is {cut_value} whereas maximal cut of {max_cut} is allowed")  # noqa: TRY002
    # print(f"{cut_value=}")

    s_idx.remove("s")
    t_idx.remove("t")
    if len(s_idx) == 0 or len(t_idx) == 0:
        raise Exception("vertices of one side are empty - this should not happen")  # noqa: TRY002
    cc_erode[tuple(np.asarray(list(s_idx)).reshape([-1, 3]).transpose())] = 1
    cc_erode[tuple(np.asarray(list(t_idx)).reshape([-1, 3]).transpose())] = 2

    lost = np.abs((cc_erode > 0).sum() - vol.sum())
    if lost > max_errors:
        raise Exception(f"lost {lost} points while separating but only {max_errors} losts allowed")  # noqa: TRY002
    return cc_erode, (S, T, G_, S_dil, T_dil, G)
    # print(cc_erode[(a + b) // 2])
