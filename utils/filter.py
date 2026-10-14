"""Multi-scale depth mismatch filtering: standalone v3 implementation.

At each scale, select local centers, sample correspondences, fit scale and
translation, and reject local outliers. Each scale reads the original input
mask independently; the final mask is the intersection of their results.

Requires CUDA PyTorch and Triton. Sampling helpers are embedded below.
This file can be copied on its own; the main entry point is multi_scale_filter_depth.
"""

import math
from itertools import chain

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = [
    "DEFAULT_FILTER_CONFIGS",
    "align_points_scale_xyz_shift",
    "filter_depth",
    "multi_scale_filter_depth",
]


# -----------------------------------------------------------------------------
# Local neighborhoods: count valid points, sample correspondences, and update masks
# -----------------------------------------------------------------------------
# AB / AI / AJ hold each center's batch, row, and column indices.
# P is the window side length; R is its pixel radius. PRED / GT are flattened
# XYZ point maps; RADIUS holds the 3D radius of each local neighborhood.


# Valid points must satisfy the input mask, image bounds, and predicted 3D sphere.
@triton.jit
def _pixel(PRED, MASK, anchor, pixel, inside, valid, radius):
    x = tl.load(PRED + pixel * 3, valid, 0.0)
    y = tl.load(PRED + pixel * 3 + 1, valid, 0.0)
    z = tl.load(PRED + pixel * 3 + 2, valid, 0.0)
    dx = x - tl.load(PRED + anchor * 3)
    dy = y - tl.load(PRED + anchor * 3 + 1)
    dz = z - tl.load(PRED + anchor * 3 + 2)
    distance = tl.sqrt((dx * dx + dy * dy) + dz * dz)
    m = valid & inside & tl.load(MASK + pixel, valid, False) & (distance <= radius)
    return x, y, z, m


# Count points in window blocks to avoid per-pixel Python loops.
@triton.jit
def count_patches(
    PRED, MASK, AB, AI, AJ, RADIUS, COUNTS,
    H: tl.constexpr, W: tl.constexpr,
    R: tl.constexpr, P: tl.constexpr, BLOCK: tl.constexpr,
):
    patch = tl.program_id(0)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = k < P * P
    batch = tl.load(AB + patch)
    ai = tl.load(AI + patch)
    aj = tl.load(AJ + patch)
    row = ai + k // P - R
    col = aj + k % P - R
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    row = tl.minimum(tl.maximum(row, 0), H - 1)
    col = tl.minimum(tl.maximum(col, 0), W - 1)
    anchor = batch * H * W + ai * W + aj
    pixel = batch * H * W + row * W + col
    radius = tl.load(RADIUS + patch)
    x, y, z, m = _pixel(PRED, MASK, anchor, pixel, inside, valid, radius)
    tl.atomic_add(COUNTS + patch, tl.sum(m.to(tl.int32), 0), sem="relaxed")


# Select the valid point nearest each sampling cell's center, using the same pixel
# in PRED and GT. N is the number of cells; F is the number of candidates per cell.
# BF / BN control GPU padding and block sizes.
@triton.jit
def sample_patches(
    PRED, GT, MASK, AB, AI, AJ, RADIUS,
    OFFSETS, DISTS, VALID, PP, GP, WEIGHT,
    H: tl.constexpr, W: tl.constexpr, R: tl.constexpr, P: tl.constexpr,
    N: tl.constexpr, F: tl.constexpr, BF: tl.constexpr, BN: tl.constexpr,
):
    patch = tl.program_id(0)
    sample = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BF)
    valid = (sample[:, None] < N) & (k[None, :] < F)
    offset = sample[:, None] * F + k[None, :]
    index = tl.load(OFFSETS + offset, valid, 0)
    dist = tl.load(DISTS + offset, valid, float("inf"))
    member = tl.load(VALID + offset, valid, False)
    batch = tl.load(AB + patch)
    ai = tl.load(AI + patch)
    aj = tl.load(AJ + patch)
    row = ai + index // P - R
    col = aj + index % P - R
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W) & member
    row = tl.minimum(tl.maximum(row, 0), H - 1)
    col = tl.minimum(tl.maximum(col, 0), W - 1)
    anchor = batch * H * W + ai * W + aj
    pixel = batch * H * W + row * W + col
    radius = tl.load(RADIUS + patch)
    x, y, z, m = _pixel(PRED, MASK, anchor, pixel, inside, valid, radius)
    cost = tl.where(m, dist, float("inf"))
    minimum = tl.min(cost, 1)
    # Break distance ties by candidate order for stable selection.
    choice = tl.min(
        tl.where((cost == minimum[:, None]) & (k[None, :] < F), k[None, :], 2147483647), 1
    )
    selected = tl.sum(tl.where(k[None, :] == choice[:, None], pixel, 0), 1)
    exists = tl.sum(m.to(tl.int32), 1) > 0
    out = patch * N + sample
    for axis in tl.static_range(3):
        pp = tl.load(PRED + selected * 3 + axis, sample < N, 0.0)
        gp = tl.load(GT + selected * 3 + axis, sample < N, 0.0)
        tl.store(PP + out * 3 + axis, pp, sample < N)
        tl.store(GP + out * 3 + axis, gp, sample < N)
    w = tl.div_rn(exists.to(tl.float32), radius + 1.0e-7)
    tl.store(WEIGHT + out, w, sample < N)


@triton.jit
def scatter_direct(
    PRED, GT, MASK, AB, AI, AJ, RADIUS, SCALE, SHIFT, OUT,
    H: tl.constexpr, W: tl.constexpr, R: tl.constexpr, P: tl.constexpr,
    RATIO: tl.constexpr, BLOCK: tl.constexpr,
):
    patch = tl.program_id(0)
    s = tl.load(SCALE + patch)
    # Skip nonpositive scales; other neighborhoods or scales can check these points.
    if s > 0:
        k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = k < P * P
        batch = tl.load(AB + patch)
        ai = tl.load(AI + patch)
        aj = tl.load(AJ + patch)
        row = ai + k // P - R
        col = aj + k % P - R
        inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        row = tl.minimum(tl.maximum(row, 0), H - 1)
        col = tl.minimum(tl.maximum(col, 0), W - 1)
        anchor = batch * H * W + ai * W + aj
        pixel = batch * H * W + row * W + col
        radius = tl.load(RADIUS + patch)
        x, y, z, m = _pixel(PRED, MASK, anchor, pixel, inside, valid, radius)
        dx = x - (s * tl.load(GT + pixel * 3, valid, 0.0) + tl.load(SHIFT + patch * 3))
        dy = y - (s * tl.load(GT + pixel * 3 + 1, valid, 0.0) + tl.load(SHIFT + patch * 3 + 1))
        dz = z - (s * tl.load(GT + pixel * 3 + 2, valid, 0.0) + tl.load(SHIFT + patch * 3 + 2))
        distance = tl.sqrt((dx * dx + dy * dy) + dz * dz)
        # Check every valid neighbor. Overlapping patches can only clear mask entries.
        tl.atomic_and(OUT + pixel, 0, m & (distance > radius * RATIO), sem="relaxed")


# -----------------------------------------------------------------------------
# Local alignment: fit s * src + t to tgt
# -----------------------------------------------------------------------------
# Each correspondence can be an alignment anchor. Fixing the anchor makes
# translation a function of scale. N is the number of correspondences, M = 3 * N,
# and B is the GPU vector length padded to a power of two.


# Recover scale and translation from the best anchor, breaking ties by index.
@triton.jit
def _reduce_anchors(
    SRC, TGT, LOSS, INDEX, OUT_S, OUT_T, N: tl.constexpr, M: tl.constexpr, B: tl.constexpr
):
    batch = tl.program_id(0)
    k = tl.arange(0, B)
    values = tl.load(LOSS + batch * N + k, k < N, float("inf"))
    minimum = tl.min(values, axis=0)
    anchor = tl.min(tl.where((k < N) & (values == minimum), k, 2147483647), axis=0)
    coordinate = tl.load(INDEX + batch * N + anchor)
    other = coordinate // 3
    axis = coordinate % 3
    p0 = tl.load(SRC + batch * M + anchor * 3 + axis)
    q0 = tl.load(TGT + batch * M + anchor * 3 + axis)
    p1 = tl.load(SRC + batch * M + coordinate)
    q1 = tl.load(TGT + batch * M + coordinate)
    scale = tl.div_rn(q1 - q0, tl.where(p1 != p0, p1 - p0, 1.0))
    tl.store(OUT_S + batch, scale)
    c = tl.arange(0, 4)
    pp = tl.load(SRC + batch * M + anchor * 3 + c, c < 3, 0.0)
    qq = tl.load(TGT + batch * M + anchor * 3 + c, c < 3, 0.0)
    tl.store(OUT_T + batch * 3 + c, qq - scale * pp, c < 3)


# Let each group choose its own scale; summing group minima gives a loss lower bound.
@triton.jit
def _group_bound(x, y, w, B: tl.constexpr, TRUNC: tl.constexpr, G: tl.constexpr):
    xx = x.reshape([G, B // G])
    yy = y.reshape([G, B // G])
    ww = w.reshape([G, B // G])
    roots = tl.div_rn(yy, tl.where(xx > 0, xx, 1.0))
    group_id = tl.arange(0, G)
    best = tl.full((B // G,), float("inf"), tl.float32)
    for k in tl.static_range(G):
        scale = tl.sum(tl.where(group_id[:, None] == k, roots, 0.0), 0)
        costs = tl.minimum(tl.abs(scale[None, :] * xx - yy) * ww, TRUNC)
        value = tl.sum(costs, 0)
        best = tl.minimum(best, value)
    return tl.sum(best, 0)


# Prune anchors whose lower bounds exceed a feasible loss, retaining all loss terms.
@triton.jit
def _prune_groups(
    SRC, TGT, W, BOUND, KEEP, LOWER,
    N: tl.constexpr, M: tl.constexpr, B: tl.constexpr,
    TRUNC: tl.constexpr, SEEDS: tl.constexpr, G: tl.constexpr,
):
    anchor = tl.program_id(0) % N
    batch = tl.program_id(0) // N
    wa = tl.load(W + batch * N + anchor)
    keep = tl.full((), False, tl.int1)
    bound = tl.full((), float("inf"), tl.float32)
    if wa > 0:
        i = tl.arange(0, B)
        dx = tl.load(SRC + batch * M + i, i < M, 0.0) - tl.load(
            SRC + batch * M + anchor * 3 + i % 3
        )
        dy = tl.load(TGT + batch * M + i, i < M, 0.0) - tl.load(
            TGT + batch * M + anchor * 3 + i % 3
        )
        w = tl.load(W + batch * N + i // 3, i < M, 0.0)
        sign = tl.where(dx > 0, 1.0, tl.where(dx < 0, -1.0, 0.0))
        bound = _group_bound(dx * sign, dy * sign, w, B, TRUNC, G)
        seed = tl.min(tl.load(BOUND + batch * SEEDS + tl.arange(0, SEEDS)), 0)
        keep = bound <= seed + 1.0e-3
    tl.store(KEEP + batch * N + anchor, keep)
    tl.store(LOWER + batch * N + anchor, bound)


# Search candidate scale intervals with branch pruning. SEED mode first solves
# a few anchors to obtain feasible upper bounds on the loss.
@triton.jit
def _branch_solver(
    SRC, TGT, W, LOSS, INDEX, NODES, BOUND, KEEP, SELECT, SEED_S,
    N: tl.constexpr, M: tl.constexpr, B: tl.constexpr,
    TRUNC: tl.constexpr, PRUNE: tl.constexpr, SEED: tl.constexpr,
    SEEDS: tl.constexpr = 1, GROUP: tl.constexpr = 0, COMPACT: tl.constexpr = False,
):
    if SEED:
        batch = tl.program_id(0) // SEEDS
        seed_slot = tl.program_id(0) % SEEDS
        j = tl.arange(0, B)
        weights = tl.load(W + batch * N + j, j < N, 0.0)
        center = (seed_slot + 1) * N // (SEEDS + 1)
        distance = tl.where((j < N) & (weights > 0), tl.abs(j - center), 2147483647)
        md = tl.min(distance, 0)
        anchor = tl.min(tl.where(distance == md, j, 2147483647), 0)
    else:
        anchor_row = tl.load(SELECT + tl.program_id(0)) if COMPACT else tl.program_id(0)
        anchor = anchor_row % N
        batch = anchor_row // N
    wa = tl.load(W + batch * N + anchor)
    if GROUP and not SEED:
        wa = tl.where(tl.load(KEEP + batch * N + anchor), wa, 0.0)
    if wa > 0:
        i = tl.arange(0, B)
        p = i // 3
        axis = i % 3
        dx = tl.load(SRC + batch * M + i, i < M, 0.0) - tl.load(SRC + batch * M + anchor * 3 + axis)
        dy = tl.load(TGT + batch * M + i, i < M, 0.0) - tl.load(TGT + batch * M + anchor * 3 + axis)
        w = tl.load(W + batch * N + p, i < M, 0.0)
        sign = tl.where(dx > 0, 1.0, tl.where(dx < 0, -1.0, 0.0))
        x = dx * sign
        y = dy * sign
        active = (i < M) & (w * x > 0)
        a = tl.div_rn(y, tl.maximum(x, 1.0e-7))
        a = tl.where(active, a, float("inf"))
        # Sort scales with the original coordinate index in the low 32 bits for ties.
        bits = a.to(tl.uint32, bitcast=True)
        key = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
        ordered = tl.sort((key.to(tl.uint64) << 32) | i.to(tl.uint64), descending=False)
        order = (ordered & 0xFFFFFFFF).to(tl.int32)
        a = tl.gather(a, order, 0)
        x = tl.gather(x, order, 0)
        y = tl.gather(y, order, 0)
        w = tl.gather(w, order, 0)
        count = tl.sum(active.to(tl.int32), 0)
        wx = x * w
        lo = tl.full((), 0, tl.int32)
        size = tl.full((), B, tl.int32)
        best = tl.full((), float("inf"), tl.float32)
        best_idx = tl.full((), 0, tl.int32)
        if PRUNE and not SEED:
            seed_values = tl.load(BOUND + batch * SEEDS + tl.arange(0, SEEDS))
            bound_seed = tl.min(seed_values, 0)
        else:
            bound_seed = tl.full((), float("inf"), tl.float32)
        nodes = tl.full((), 0, tl.int32)
        while lo < B:
            descend = tl.full((), False, tl.int1)
            if lo < count:
                nodes += 1
                hi = tl.minimum(lo + size - 1, count - 1)
                left = tl.sum(tl.where(i == lo, a, 0.0), 0)
                right = tl.sum(tl.where(i == hi, a, 0.0), 0)
                rl = tl.abs(left * x - y) * w
                rr = tl.abs(right * x - y) * w
                # Epsilon shifts roots for tiny x; use zero as a safe term lower bound.
                lowest = tl.where(
                    (x < 1.0e-7) | ((a >= left) & (a <= right)), 0.0, tl.minimum(rl, rr)
                )
                easy_bound = tl.sum(tl.minimum(lowest, TRUNC), 0)
                # Allow a margin for float32 rounding at the pruning boundary.
                if easy_bound <= tl.minimum(best, bound_seed) + 1.0e-4:
                    convex = (tl.maximum(rl, rr) <= TRUNC) & (i < count) & (x >= 1.0e-7)
                    flat = (lowest >= TRUNC) | (i >= count)
                    cw = tl.where(convex, wx, 0.0)
                    total = tl.sum(cw, 0)
                    if total > 0:
                        prefix = tl.cumsum(cw, 0)
                        median = tl.min(
                            tl.where((i < count) & (prefix >= total * 0.5), i, 2147483647), 0
                        )
                        mid = tl.minimum(tl.maximum(median, lo), hi)
                    else:
                        mid = (lo + hi) // 2
                    pivot = tl.sum(tl.where(i == mid, a, 0.0), 0)
                    residual = tl.minimum(tl.abs(pivot * x - y) * w, TRUNC)
                    value = tl.sum(residual, 0)
                    index = tl.sum(tl.where(i == mid, order, 0), 0)
                    better = (value < best) | ((value == best) & (index < best_idx))
                    best = tl.where(better, value, best)
                    best_idx = tl.where(better, index, best_idx)
                    bound = tl.sum(tl.where(convex, residual, tl.minimum(lowest, TRUNC)), 0)
                    crossing = tl.sum((~(convex | flat)).to(tl.int32), 0) > 0
                    descend = (
                        (size > 1)
                        & (lo < hi)
                        & crossing
                        & (bound <= tl.minimum(best, bound_seed) + 1.0e-4)
                    )
            if descend:
                size = size // 2
            else:
                lo += size
                while (size < B) & ((lo % (size * 2)) == 0):
                    size *= 2
        best = tl.where(count == 0, 0.0, best)
        if SEED:
            tl.store(BOUND + batch * SEEDS + seed_slot, best)
            sx = tl.load(SRC + batch * M + best_idx) - tl.load(
                SRC + batch * M + anchor * 3 + best_idx % 3
            )
            sy = tl.load(TGT + batch * M + best_idx) - tl.load(
                TGT + batch * M + anchor * 3 + best_idx % 3
            )
            sign_sx = tl.where(sx > 0, 1.0, tl.where(sx < 0, -1.0, 0.0))
            ss = tl.div_rn(sy * sign_sx, tl.maximum(sx * sign_sx, 1.0e-7))
            tl.store(SEED_S + batch * SEEDS + seed_slot, ss)
        else:
            tl.store(LOSS + batch * N + anchor, best)
            tl.store(INDEX + batch * N + anchor, best_idx)
            tl.store(NODES + batch * N + anchor, nodes)
    else:
        if SEED:
            tl.store(BOUND + batch * SEEDS + seed_slot, float("inf"))
            tl.store(SEED_S + batch * SEEDS + seed_slot, 0.0)
        else:
            tl.store(LOSS + batch * N + anchor, float("inf"))
            tl.store(INDEX + batch * N + anchor, 0)
            tl.store(NODES + batch * N + anchor, 0)


# Snap seed scales to each anchor's candidate roots and evaluate the full loss
# to tighten the upper bound.
@triton.jit
def _tighten_bound(
    SRC, TGT, W, BOUND, SEED_S, KEEP, UPPER,
    N: tl.constexpr, M: tl.constexpr, B: tl.constexpr,
    SEEDS: tl.constexpr, TOP: tl.constexpr, TRUNC: tl.constexpr,
):
    anchor = tl.program_id(0) % N
    batch = tl.program_id(0) // N
    if tl.load(KEEP + batch * N + anchor):
        i = tl.arange(0, B)
        dx = tl.load(SRC + batch * M + i, i < M, 0.0) - tl.load(
            SRC + batch * M + anchor * 3 + i % 3
        )
        dy = tl.load(TGT + batch * M + i, i < M, 0.0) - tl.load(
            TGT + batch * M + anchor * 3 + i % 3
        )
        w = tl.load(W + batch * N + i // 3, i < M, 0.0)
        sign = tl.where(dx > 0, 1.0, tl.where(dx < 0, -1.0, 0.0))
        x = dx * sign
        y = dy * sign
        active = (i < M) & (w * x > 0)
        roots = tl.div_rn(y, tl.maximum(x, 1.0e-7))
        slots = tl.arange(0, SEEDS)
        losses = tl.load(BOUND + batch * SEEDS + slots)
        best = tl.min(losses, 0)
        for t in tl.static_range(TOP):
            slot = tl.min(tl.where(losses == tl.min(losses, 0), slots, 2147483647), 0)
            estimate = tl.load(SEED_S + batch * SEEDS + slot)
            delta = tl.where(active, tl.abs(roots - estimate), float("inf"))
            candidate = tl.min(tl.where(delta == tl.min(delta, 0), i, 2147483647), 0)
            pivot = tl.sum(tl.where(i == candidate, roots, 0.0), 0)
            loss = tl.sum(tl.minimum(tl.abs(pivot * x - y) * w, TRUNC), 0)
            best = tl.minimum(best, loss)
            losses = tl.where(slots == slot, float("inf"), losses)
        if best < tl.load(UPPER + batch):
            tl.atomic_min(UPPER + batch, best, sem="relaxed")


def align_points_branch(
    points_src, points_tgt, weight, trunc=1.0, *,
    warps=4, prune=False, seeds=1, group=0, fuse=False, compact=False, tighten=1,
):
    """Fit local scale and translation using upper and lower bounds to prune search.

    Points have shape (..., N, 3), and weights have shape (..., N). The objective
    sums weighted L1 residuals, clipping each coordinate's contribution at trunc.
    Return scales of shape (...) and translations of shape (..., 3).
    """
    if group and not prune:
        raise ValueError("Group pruning requires feasible seed bounds")
    if group and not 0 <= tighten <= seeds:
        raise ValueError("tighten must be between zero and the number of seeds")
    shape = points_src.shape[:-2]
    n = points_src.shape[-2]
    src = points_src.reshape(-1, n, 3).contiguous()
    tgt = points_tgt.reshape(-1, n, 3).contiguous()
    weights = weight.reshape(-1, n).contiguous()
    batch = src.shape[0]
    loss = (
        torch.full((batch, n), float("inf"), device=src.device)
        if compact
        else torch.empty((batch, n), device=src.device)
    )
    index = (
        torch.zeros((batch, n), device=src.device, dtype=torch.int32)
        if compact
        else torch.empty((batch, n), device=src.device, dtype=torch.int32)
    )
    nodes = torch.empty_like(index)
    s = torch.empty((batch,), device=src.device)
    t = torch.empty((batch, 3), device=src.device)
    bound = torch.full((batch, seeds), float("inf"), device=src.device)
    keep = torch.empty((batch, n), device=src.device, dtype=torch.bool)
    lower = torch.empty((batch, n), device=src.device)
    seed_scale = torch.empty_like(bound)
    # Solve seed anchors, then use group lower bounds to prune remaining anchors.
    if prune:
        _branch_solver[(batch * seeds,)](
            src, tgt, weights, loss, index, nodes, bound, keep, keep, seed_scale,
            N=n, M=3 * n, B=triton.next_power_of_2(3 * n),
            TRUNC=float(trunc), PRUNE=True, SEED=True, SEEDS=seeds,
            num_warps=warps, enable_fp_fusion=fuse,
        )
    if group:
        _prune_groups[(n * batch,)](
            src, tgt, weights, bound, keep, lower,
            N=n, M=3 * n, B=triton.next_power_of_2(3 * n),
            TRUNC=float(trunc), SEEDS=seeds, G=group,
            num_warps=1 if n <= 144 else 2, enable_fp_fusion=fuse,
        )
    tightened = bound.min(-1).values
    # Tighten the upper bound using roots near seed scales for retained anchors.
    if tighten and group:
        _tighten_bound[(n * batch,)](
            src, tgt, weights, bound, seed_scale, keep, tightened,
            N=n, M=3 * n, B=triton.next_power_of_2(3 * n),
            SEEDS=seeds, TOP=tighten, TRUNC=float(trunc),
            num_warps=1, enable_fp_fusion=fuse,
        )
        keep = (weights > 0) & (lower <= tightened[:, None] + 1.0e-3)
    # Search retained anchors; every loss still includes all valid correspondences.
    selected = torch.where(keep.flatten())[0] if compact and group else None
    _branch_solver[(selected.numel() if selected is not None else n * batch,)](
        src, tgt, weights, loss, index, nodes, tightened, keep,
        selected if selected is not None else keep, seed_scale,
        N=n, M=3 * n, B=triton.next_power_of_2(3 * n),
        TRUNC=float(trunc), PRUNE=prune, SEED=False, SEEDS=1,
        GROUP=group, COMPACT=selected is not None,
        num_warps=warps, enable_fp_fusion=fuse,
    )
    # Select the best anchor and return s and t for each local neighborhood.
    _reduce_anchors[(batch,)](
        src, tgt, loss, index, s, t,
        N=n, M=3 * n, B=triton.next_power_of_2(n),
        num_warps=1, enable_fp_fusion=False,
    )
    return s.reshape(shape), t.reshape(*shape, 3)


# -----------------------------------------------------------------------------
# Pack correspondences: remove zero weights while preserving order and duplicates
# -----------------------------------------------------------------------------


@triton.jit
def pack_points(
    SRC, TGT, W, SELECT, OUT_SRC, OUT_TGT, OUT_W,
    N: tl.constexpr, CAP: tl.constexpr, B: tl.constexpr,
):
    row = tl.program_id(0)
    old = tl.load(SELECT + row)
    i = tl.arange(0, B)
    w = tl.load(W + old * N + i, i < N, 0.0)
    valid = (i < N) & (w > 0)
    dest = tl.cumsum(valid.to(tl.int32), 0) - 1
    # Prefix sums preserve order. Outputs start at zero, including padding weights.
    tl.store(OUT_W + row * CAP + dest, w, valid)
    for axis in tl.static_range(3):
        x = tl.load(SRC + (old * N + i) * 3 + axis, valid, 0.0)
        y = tl.load(TGT + (old * N + i) * 3 + axis, valid, 0.0)
        tl.store(OUT_SRC + (row * CAP + dest) * 3 + axis, x, valid)
        tl.store(OUT_TGT + (row * CAP + dest) * 3 + axis, y, valid)


def align_points(src, tgt, weight, trunc=1.0, *, minimum=36):
    """Bucket by valid count to reduce padding; preserve all points and duplicates."""
    shape = src.shape[:-2]
    n = src.shape[-2]
    if n <= minimum:
        return align_points_branch(
            src, tgt, weight, trunc, prune=True,
            seeds=4 if n > 36 else 1, group=32 if n > 36 else 0, warps=4 if n > 36 else 1,
            fuse=n > 36, compact=n > 36, tighten=2 if n > 36 else 0,
        )
    src = src.reshape(-1, n, 3).contiguous()
    tgt = tgt.reshape(-1, n, 3).contiguous()
    weight = weight.reshape(-1, n).contiguous()
    batch = src.shape[0]
    counts = (weight > 0).sum(-1)
    scale = torch.zeros(batch, device=src.device, dtype=src.dtype)
    shift = torch.zeros((batch, 3), device=src.device, dtype=src.dtype)
    # Keep 3 * cap close to a power of two to reduce padding of XYZ vectors.
    capacities = sorted(set([c for c in (21, 42, 85, 170, 341) if c < n] + [n]))
    lower = 0
    for cap in capacities:
        selected = torch.where((counts > lower) & (counts <= cap))[0]
        lower = cap
        if selected.numel() == 0:
            continue
        a = torch.zeros((selected.numel(), cap, 3), device=src.device, dtype=src.dtype)
        b = torch.zeros_like(a)
        w = torch.zeros((selected.numel(), cap), device=src.device, dtype=weight.dtype)
        pack_points[(selected.numel(),)](
            src, tgt, weight, selected, a, b, w,
            N=n, CAP=cap, B=triton.next_power_of_2(n),
            num_warps=4 if n > 144 else 1,
        )
        s, t = align_points_branch(
            a, b, w, trunc, prune=True,
            seeds=4 if n > 36 else 1, group=32 if n > 36 else 0, warps=4 if n > 36 else 1,
            fuse=n > 36, compact=n > 36, tighten=2 if n > 36 else 0,
        )
        scale[selected] = s
        shift[selected] = t
    return scale.reshape(shape), shift.reshape(*shape, 3)


_align_points = align_points


# -----------------------------------------------------------------------------
# Sampling templates and public API
# -----------------------------------------------------------------------------

# The three helpers below are copied from utils3d, with shorter documentation
# and type annotations omitted. Their computation and defaults are unchanged.
# Source: https://github.com/EasternJournalist/utils3d
# Commit: 3fab839f0be9931dac7c8488eb0e1600c236e183
# Files: utils3d/torch/maps.py and utils3d/torch/utils.py
#
# MIT License
#
# Copyright (c) 2022 EasternJournalist
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def uv_map(*size, top=0.0, left=0.0, bottom=1.0, right=1.0, dtype=torch.float32, device=None):
    """Return pixel-center UV coordinates of shape (height, width, 2)."""
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.linspace(left + 0.5 / width, right - 0.5 / width, width, dtype=dtype, device=device)
    v = torch.linspace(
        top + 0.5 / height, bottom - 0.5 / height, height, dtype=dtype, device=device
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=2)


def pixel_coord_map(
    *size, top=0, left=0, convention="integer-center", dtype=torch.float32, device=None,
):
    """Return XY coordinates; integer-corner convention adds 0.5 to each axis."""
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.arange(left, left + width, dtype=dtype, device=device)
    v = torch.arange(top, top + height, dtype=dtype, device=device)
    if convention == "integer-corner":
        assert torch.is_floating_point(u), (
            "dtype should be a floating point type when convention is 'integer-corner'"
        )
        u = u + 0.5
        v = v + 0.5
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=2)


def sliding_window(
    x, window_size, stride=None, pad_size=None, pad_mode="constant", pad_value=0, dim=None,
):
    """Pad selected dimensions and append window axes using Tensor.unfold."""
    if dim is None:
        dim = tuple(range(x.ndim))
    if isinstance(dim, int):
        dim = (dim,)
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    if isinstance(window_size, int):
        window_size = (window_size,) * len(dim)
    if stride is None:
        stride = (1,) * len(dim)
    if isinstance(stride, int):
        stride = (stride,) * len(dim)
    assert len(window_size) == len(stride) == len(dim)

    if pad_size is not None:
        if isinstance(pad_size, int):
            pad_size = ((pad_size, pad_size),) * len(dim)
        elif (
            isinstance(pad_size, tuple) and len(pad_size) == 2
            and all(isinstance(p, int) for p in pad_size)
        ):
            pad_size = (pad_size,) * len(dim)
        elif isinstance(pad_size, tuple) and all(
            isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size
        ):
            if len(pad_size) == 1:
                pad_size = pad_size * len(dim)
            else:
                assert len(pad_size) == len(dim), (
                    f"pad_size {pad_size} must match the number of axes {len(dim)}"
                )
            pad_size = tuple(p * 2 if len(p) == 1 else p for p in pad_size)
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)] for i in range(x.ndim)]
        full_pad = tuple(chain(*reversed(full_pad)))
        x = F.pad(x, full_pad, mode=pad_mode, value=pad_value)

    for i in range(len(window_size)):
        x = x.unfold(dim[i], window_size[i], stride[i])
    return x


# Each entry is (level, align_resolution, K, chunk_size).
DEFAULT_FILTER_CONFIGS = ((4, 24, 64, 4096), (16, 12, 2, 65536), (64, 6, 1, 262144))

# Cache geometry by window size, sampling resolution, and device.
_templates = {}


def sampling_template(side, res, device):
    """Return offsets, squared center distances, and validity for res * res cells.

    All three tensors have shape (res * res, candidates_per_cell). Pixel offsets
    are relative to the flattened side * side window.
    """
    key = (side, res, device)
    if key not in _templates:
        # Preserve the pinned utils3d coordinate and rounding conventions.
        width = height = side
        f = max(1, side / res)
        fi = math.ceil(f)
        padding = fi // 2 + 1
        pixels = pixel_coord_map(
            height, width, convention="integer-corner", dtype=torch.float32, device=device
        )
        index = torch.arange(height * width, device=device, dtype=torch.long).reshape(height, width)
        valid = torch.ones((height, width), device=device, dtype=torch.bool)
        kw = dict(window_size=(fi, fi), pad_size=((padding, padding), (padding, padding)))
        wp = sliding_window(pixels, **kw, dim=(0, 1))
        wi = sliding_window(index, **kw, dim=(0, 1))
        wv = sliding_window(valid, **kw, dim=(0, 1))
        centers = uv_map(res, res, dtype=torch.float32, device=device) * torch.tensor(
            [width, height], dtype=torch.float32, device=device
        )
        left = centers - torch.tensor((f / 2, f / 2), dtype=torch.float32, device=device)
        windows = torch.round(left).long() + torch.tensor(
            (padding, padding), dtype=torch.long, device=device
        )
        twp = wp[windows[..., 1], windows[..., 0], :, :, :].reshape(res, res, 2, fi * fi)
        offsets = (
            wi[windows[..., 1], windows[..., 0], :, :]
            .reshape(res * res, fi * fi)
            .int()
            .contiguous()
        )
        members = (
            wv[windows[..., 1], windows[..., 0], :, :].reshape(res * res, fi * fi).contiguous()
        )
        distance = (twp - centers[..., None]).square()
        distance = (
            (distance[..., 0, :] + distance[..., 1, :]).reshape(res * res, fi * fi).contiguous()
        )
        _templates[key] = (offsets, distance, members)
    return _templates[key]


def align_points_scale_xyz_shift(src, tgt, weight, trunc=1.0):
    """Fit s * src + t to tgt using a coordinate-wise truncated weighted L1 loss.

    Args:
        src, tgt: Corresponding CUDA float32 points of shape (..., N, 3).
        weight: Nonnegative weights of shape (..., N); zero excludes a point.
        trunc: Threshold for each coordinate's weighted residual.

    Returns:
        Scale s of shape (...) and translation t of shape (..., 3).
    """
    # Small batches at the finest scale cost more to bucket than they save.
    n = src.shape[-2]
    minimum = 0 if src.numel() // (3 * n) >= 4096 else 36
    return _align_points(src, tgt, weight, trunc, minimum=minimum)


def filter_depth(
    points_pred, points_gt, mask, focal,
    level=16, chunk_size=256, align_resolution=12, K=4, trunc=1.0, ratio=1.0,
):
    """Return a single-scale boolean keep mask, or None if the input mask is empty.

    Point maps have shape (..., H, W, 3); mask has shape (..., H, W). The normalized
    focal length tensor can provide one value per image or a single broadcast value.
    Point maps and focal lengths are converted to float32 before processing.

    Larger level values produce smaller neighborhoods. Select every K-th valid
    pixel as a local center and process up to chunk_size centers per batch.
    Sample each neighborhood on an align_resolution * align_resolution grid,
    then align GT points to predicted points.

    Check all valid neighbors after alignment, rejecting points with residuals
    above ratio times the 3D radius. The alignment loss is clipped at trunc;
    nonpositive fitted scales skip the local outlier decision.
    """
    *batch_shape, height, width, _ = points_pred.shape
    pred = points_pred.float().reshape(-1, height, width, 3).contiguous()
    gt = points_gt.float().reshape(-1, height, width, 3).contiguous()
    mask = mask.reshape(-1, height, width).contiguous()
    focal = focal.float().reshape(-1)
    if focal.numel() == 1:
        focal = focal.expand(pred.shape[0])
    anchors = tuple(x[::K] for x in torch.where(mask))
    count = anchors[0].numel()
    if count == 0:
        return
    # radius is the image window's pixel radius; radii below holds per-center 3D radii.
    radius = math.ceil(0.5 / level * math.sqrt(height**2 + width**2))
    side = radius * 2 + 1
    offsets, distances, members = sampling_template(side, align_resolution, pred.device)
    f = offsets.shape[1]
    n = align_resolution**2
    out = mask.to(torch.int32)
    for start in range(0, count, chunk_size):
        ab, ai, aj = (x[start : start + chunk_size].contiguous() for x in anchors)
        radii = 0.5 / level / focal[ab] * pred[ab, ai, aj, 2]

        # 1. Count valid points in each 3D sphere; skip patches with fewer than 32.
        counts = torch.zeros(ab.numel(), device=pred.device, dtype=torch.int32)
        block = 128 if side <= 11 else 256
        count_patches[(ab.numel(), triton.cdiv(side * side, block))](
            pred, mask, ab, ai, aj, radii, counts,
            H=height, W=width, R=radius, P=side, BLOCK=block,
            num_warps=1 if side <= 11 else 4, enable_fp_fusion=False,
        )
        keep = torch.where(counts >= 32)[0]
        if keep.numel() == 0:
            continue
        ab, ai, aj, radii = (x[keep] for x in (ab, ai, aj, radii))

        # 2. Select one valid pair per sampling cell; empty cells get weight 0.
        pp = torch.empty((keep.numel(), n, 3), device=pred.device, dtype=torch.float32)
        gp = torch.empty_like(pp)
        weight = torch.empty((keep.numel(), n), device=pred.device, dtype=torch.float32)
        bn = 4
        sample_patches[(keep.numel(), triton.cdiv(n, bn))](
            pred, gt, mask, ab, ai, aj, radii,
            offsets, distances, members, pp, gp, weight,
            H=height, W=width, R=radius, P=side,
            N=n, F=f, BF=triton.next_power_of_2(f), BN=bn,
            num_warps=4, enable_fp_fusion=False,
        )
        # 3. Fit scale * GT + shift to PRED.
        scale, shift = align_points_scale_xyz_shift(gp, pp, weight, trunc=trunc)

        # 4. Apply the alignment to every valid neighbor and update the mask.
        scatter_direct[(keep.numel(), triton.cdiv(side * side, 256))](
            pred, gt, mask, ab, ai, aj, radii, scale, shift, out,
            H=height, W=width, R=radius, P=side, RATIO=float(ratio), BLOCK=256,
            num_warps=4, enable_fp_fusion=False,
        )
    return out.bool().reshape(*batch_shape, height, width)


@torch.no_grad()
def multi_scale_filter_depth(
    points_pred, points_gt, mask, focal, trunc=1.0, scale_configs=DEFAULT_FILTER_CONFIGS
):
    """Filter depth mismatches at multiple scales and return a boolean keep mask.

    Args:
        points_pred: Predicted CUDA XYZ map, shape (..., H, W, 3).
        points_gt: XYZ map to align, with the same shape and device as points_pred.
        mask: Boolean valid-point mask, shape (..., H, W), on the same device.
        focal: Normalized focal length tensor on the same device; one value per
            image or a single value to broadcast.
        trunc: Threshold for coordinate-wise weighted L1 alignment residuals.
        scale_configs: Sequence of (level, align_resolution, K, chunk_size) tuples.
            Defaults to levels 4, 16, and 64.

    Returns:
        A boolean mask with the same shape as the input mask. Each scale reads
        the original mask independently; the result is their intersection.

    Point maps and focal lengths are converted to float32 once before all scales.
    """
    if not points_pred.is_cuda or points_gt.device != points_pred.device:
        raise ValueError("The accelerated backend requires point maps on the same CUDA device")
    # float() reuses float32 tensors; other dtypes are converted before filtering.
    points_pred = points_pred.float()
    points_gt = points_gt.float()
    focal = focal.float()
    result = mask.clone()
    for level, res, k, chunk in scale_configs:
        # Use the original mask so earlier scales cannot affect sampling or alignment.
        m = filter_depth(
            points_pred, points_gt, mask, focal,
            level=level, align_resolution=res, K=k, chunk_size=chunk, trunc=trunc,
        )
        if m is not None:
            result = result & m
    return result
