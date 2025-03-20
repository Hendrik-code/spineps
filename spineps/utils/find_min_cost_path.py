import sys
from collections import Counter
from warnings import warn

import numpy as np


def argmin(lst):
    m = min(lst)
    return lst.index(m), m


def softmax_T(x, temp):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(np.divide(x, temp)) / np.sum(np.exp(np.divide(x, temp)), axis=0)


def c_to_region_idx(c: int, regions: list[int]):
    for idx, r in enumerate(regions):
        if c < r:
            return idx - 1
    return len(regions) - 1


def internal_to_real_path(p):
    pat = sorted(p, key=lambda x: x[0])
    pat = [x[1] for x in pat]
    return pat


def find_most_probably_sequence(  # noqa: C901
    cost: np.ndarray | list[int],
    #
    min_start_class: int = 0,
    region_rel_cost: np.ndarray | list[int] | None = None,
    vertt13_cost: np.ndarray | list[int] | None = None,
    regions: list[int] | None = None,
    #
    invert_cost: bool = True,
    #
    softmax_cost: bool = False,
    softmax_temp: float = 0.2,
    #
    allow_multiple_at_class: list[int] | None = None,  # T12 and L5
    punish_multiple_sequence: float = 0.0,
    #
    allow_skip_at_class: list[int] | None = None,  # T11
    punish_skip_sequence: float = 0.0,
    #
    allow_skip_at_region: list[int] | None = None,
    punish_skip_at_region_sequence: float = 0.2,
) -> tuple[float, list[int]]:
    # default mutable arguments
    if allow_skip_at_region is None:
        allow_skip_at_region = [0]
    if allow_skip_at_class is None:
        allow_skip_at_class = [17]
    if allow_multiple_at_class is None:
        allow_multiple_at_class = [18, 23]
    if regions is None:
        regions = [0, 7, 19]
    # convert to np arrays
    if isinstance(cost, list):
        cost = np.asarray(cost)
    if region_rel_cost is not None and isinstance(region_rel_cost, list):
        region_rel_cost = np.asanyarray(region_rel_cost)
    if vertt13_cost is not None and isinstance(vertt13_cost, list):
        vertt13_cost = np.asanyarray(vertt13_cost)
    # safety assert
    assert isinstance(cost, np.ndarray)
    shape = cost.shape

    # define regions
    n_classes = shape[1]
    assert min_start_class < n_classes
    regions_ranges = None
    if region_rel_cost is not None:
        if n_classes < regions[-1]:
            warn(f"n_classes < defined regions, got {n_classes} and {regions}", stacklevel=3)
        regions.append(n_classes)
        regions_ranges = [(regions[i], regions[i + 1] - 1) for i in range(len(regions) - 1)]
        region_rel_shape = region_rel_cost.shape
        assert region_rel_shape[1] == (
            (len(regions) - 1) * 2
        ), f"expected region_rel_cost with shape {((len(regions) - 1) * 2)}, but got {region_rel_shape[1]}"

    # softmax (deprecated, handled elsewhere)
    if softmax_cost:
        cost = softmax_T(cost, softmax_temp)
    # invert cost so high numbers are actually preferred instead of repelled
    if invert_cost:
        cost = -cost

    # make costs a list
    costlist = cost.tolist()
    # init memory
    min_costs_path = [[(None, None) for y in range(shape[1])] for x in range(shape[0])]

    # Adds edges with a extra cost beyond the cost matrix
    def add_option_path(options, r, c, extracost):
        options.append(minCostAlgo(r, c))
        options[-1] = (
            options[-1][0] + extracost,
            options[-1][1],
        )
        return options

    # main recursive loop
    def minCostAlgo(r, c):
        # get current region
        region_cur = c_to_region_idx(c, regions)
        # start point
        if c == -1 and r == -1:
            # go over each possible start column
            options = []
            for cc in range(min_start_class, n_classes):
                add_option_path(options, 0, cc, 0)
                # options.append(minCostAlgo(r=0, c=cc))
            minidx, minval = argmin([o[0] for o in options])
            return minval, options[minidx][1]
        # stepped over the line
        elif c < 0 or r < 0 or c >= shape[1] or r >= shape[0]:
            return sys.maxsize, [(r, c)]
        # last row, path end
        elif r == shape[0] - 1:
            # path_tothis.append((r, c))
            cost_value = costlist[r][c]
            p = [(r, c)]
            # transition cost of vertrel
            cost_value += rel_cost(r, c, p, region_cur)
            # if cost_value < 0:
            #    print(f"Endpoint {r}, {c} to {cost_value}, {p}")
            return (cost_value, p)
        # check min of move directions
        else:
            if min_costs_path[r][c][0] is not None:
                return min_costs_path[r][c]

            # rel_costadd = rel_cost(r, c, [(r, c)], region_cur)
            options = []
            # normal diagonal edge
            add_option_path(options, r + 1, c + 1, 0)
            # allow two subsequent of same class
            if c in allow_multiple_at_class:
                cost_add = punish_multiple_sequence
                if c == 18:
                    cost_add += t13_cost_single(r + 1, c)
                add_option_path(options, r + 1, c, cost_add)
            # Allow skips at certain classes
            if c in allow_skip_at_class:
                cost_add = punish_skip_sequence
                add_option_path(options, r + 1, c + 2, cost_add)
            # Allow skips in certain regions
            if region_cur in allow_skip_at_region and c != regions_ranges[region_cur][1] - 1:
                cost_add = punish_skip_at_region_sequence
                add_option_path(options, r + 1, c + 2, punish_skip_at_region_sequence)
            # find min
            minidx, minval = argmin([o[0] for o in options])
            pnext = options[minidx][1]
            p = [*pnext, (r, c)]
            cnt = Counter([l[1] for l in p])
            #
            cost_value = minval + costlist[r][c]
            # transition cost of vertrel
            cost_value += rel_cost(r, c, pnext, region_cur)
            # constraint: cannot have more than 2 T12 and L5
            for amac in allow_multiple_at_class:
                if amac in cnt and cnt[amac] > 2:
                    cost_value = sys.maxsize
                    break
            # setting to memory
            min_costs_path[r][c] = (cost_value, p)
            # if cost_value < 0:
            #    print(f"Setting {r}, {c} to {cost_value}, {p}")
            return min_costs_path[r][c]

    # def t13_cost(r, c, pnext, p, region_cur):
    #    cost_add = 0
    #    if vertt13_cost is not None:
    #        vt13_cost = vertt13_cost[r][1]
    #        # print(r, c, p[-1][1], p[-2][1], internal_to_real_path(p))
    #        if p[-1][1] == 18 and p[-2][1] == 18:
    #            print(f"Added F {vt13_cost} to {r}, {c}, {internal_to_real_path(p)}")
    #            cost_add += vt13_cost
    #    return cost_add

    def t13_cost_single(r, c):
        cost_add = 0
        if vertt13_cost is not None:
            vt13_cost = vertt13_cost[r][1]
            if c == 18:
                # print(f"Added F {vt13_cost} to {r}, {c}")
                cost_add += vt13_cost
        return cost_add

    def rel_cost(r, c, pnext, region_cur):
        # transition cost of vertrel
        # first is just equal to that specific vertebra
        # last is dependant on next in path
        # classes are always first, last in order of regions
        cost_add = 0
        if region_rel_cost is not None:
            # for ridx in range(len(regions) - 1):
            for last in [0, 1]:
                if region_cur + last == 0:
                    continue
                region_cls = (region_cur * 2) + last  # 0 is nothing
                rel_cost = region_rel_cost[r][region_cls]
                if rel_cost == 0:
                    continue
                if last == 0 and c == regions_ranges[region_cur][0]:
                    # print(f"Added F {rel_cost} to {r}, {c}, {internal_to_real_path(pnext)}")
                    cost_add += rel_cost
                    # break
                elif last == 1 and c_to_region_idx(pnext[-1][1], regions) >= region_cur + 1:
                    # print(f"Added L {rel_cost} to {r}, {c}, {internal_to_real_path(pnext)}")
                    cost_add += rel_cost
        return cost_add

    fcost, fpath = minCostAlgo(-1, -1)
    fpath.reverse()
    fpath = [f[1] for f in fpath]
    return fcost, fpath, min_costs_path
