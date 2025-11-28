"""
obamify.py
Simple implementation of the 'obamify' idea:
 - divide source and target into RESOLUTION x RESOLUTION cells
 - compute a cost matrix between source cells and target cells:
     cost = color_weight * color_dist + spatial_weight * spatial_dist
 - solve an assignment (Hungarian / greedy) and produce a new image
"""
from PIL import Image
import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse
import math
import sys

def image_to_cells(img: Image.Image, resolution: int):
    """Resize to square resolution*resolution cells and return arrays:
       colors: (N,3) average RGB per cell
       centers: (N,2) normalized center coordinates (0..1)
       cell_pixels: list of arrays for reconstructing if desired (not used)
    """
    img = img.convert("RGB")
    size = (resolution, resolution)
    small = img.resize(size, Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)  # shape (res, res, 3)
    res = resolution
    colors = arr.reshape((-1, 3))  # row-major
    # centers: x,y normalized [0,1]
    ys, xs = np.indices((res, res))
    xs = xs.reshape(-1) / (res - 1) if res > 1 else xs.reshape(-1) * 0.0
    ys = ys.reshape(-1) / (res - 1) if res > 1 else ys.reshape(-1) * 0.0
    centers = np.stack([xs, ys], axis=1)
    return colors, centers

def build_cost_matrix(src_colors, src_centers, tgt_colors, tgt_centers,
                      color_weight=1.0, spatial_weight=0.5):
    """Return an (N,N) cost matrix where N = number of cells."""
    # color distances squared
    # shape (N,1,3) and (1,N,3) broadcasting to (N,N,3)
    sc = src_colors[:, None, :]  # (N,1,3)
    tc = tgt_colors[None, :, :]  # (1,N,3)
    color_diff = sc - tc  # (N,N,3)
    color_cost = np.sum(color_diff * color_diff, axis=2)  # (N,N)

    # spatial distances squared
    ss = src_centers[:, None, :]  # (N,1,2)
    ts = tgt_centers[None, :, :]  # (1,N,2)
    spatial_diff = ss - ts
    spatial_cost = np.sum(spatial_diff * spatial_diff, axis=2)  # (N,N)

    # normalize each cost to comparable ranges (optional but helpful)
    color_cost = color_cost / (255.0**2 * 3.0)  # roughly 0..1
    # spatial_cost already in 0..2 (if centers in 0..1). normalize to 0..1
    spatial_cost = spatial_cost / 2.0

    cost = color_weight * color_cost + spatial_weight * spatial_cost
    return cost

def assign_optimal(cost_matrix):
    """Solve assignment using Hungarian algorithm (scipy)."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind are source indices, col_ind are target indices assigned to them
    return row_ind, col_ind

def assign_greedy(cost_matrix):
    """Simple O(N^2) greedy assignment: repeatedly pick smallest cost and lock pair.
       Not optimal but fast and memory-friendly for moderate sizes.
    """
    N = cost_matrix.shape[0]
    assigned_src = np.zeros(N, dtype=bool)
    assigned_tgt = np.zeros(N, dtype=bool)
    src_to_tgt = -np.ones(N, dtype=int)

    # flatten indices sorted by cost
    # careful with memory: if N>4096, flattening NxN might be big, but still ok for N<=4096
    flat_idx = np.argsort(cost_matrix.ravel())
    for f in flat_idx:
        s = f // N
        t = f % N
        if (not assigned_src[s]) and (not assigned_tgt[t]):
            assigned_src[s] = True
            assigned_tgt[t] = True
            src_to_tgt[s] = t
        if assigned_src.all():
            break
    return np.arange(N), src_to_tgt

def render_output(src_img: Image.Image, tgt_img: Image.Image,
                  src_to_tgt, resolution):
    """
    Construct output image (same size as src_img) by mapping each src cell
    to color of assigned target cell. Output is a new image sized to src_img.
    """
    # Resize to resolution to create the mosaic then scale back
    src_small = src_img.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    tgt_small = tgt_img.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    src_pixels = np.array(src_small).reshape((-1, 3))
    tgt_pixels = np.array(tgt_small).reshape((-1, 3))
    N = resolution * resolution
    out_pixels = np.zeros_like(src_pixels)
    # src_to_tgt: for each source index, which target index we use
    for s in range(N):
        t = int(src_to_tgt[s])
        if t < 0:
            out_pixels[s] = src_pixels[s]  # fallback to original
        else:
            out_pixels[s] = tgt_pixels[t]
    out_small = out_pixels.reshape((resolution, resolution, 3)).astype(np.uint8)
    out_img = Image.fromarray(out_small, mode="RGB")
    # scale back to original src size (or to target size if you prefer)
    out_full = out_img.resize(src_img.size, Image.NEAREST)
    return out_full

def obamify(source_path, target_path, out_path,
            resolution=64, proximity=0.5, color_weight=1.0, algorithm='optimal'):
    src = Image.open(source_path)
    tgt = Image.open(target_path)
    # make both square by cropping center to min dimension
    min_side = min(src.size[0], src.size[1], tgt.size[0], tgt.size[1])
    # center-crop helper
    def center_crop(im, side):
        w, h = im.size
        left = (w - side)//2
        top = (h - side)//2
        return im.crop((left, top, left+side, top+side))
    srcc = center_crop(src, min_side)
    tgtc = center_crop(tgt, min_side)

    # convert to cell features
    src_colors, src_centers = image_to_cells(srcc, resolution)
    tgt_colors, tgt_centers = image_to_cells(tgtc, resolution)

    # build cost matrix
    cost = build_cost_matrix(src_colors, src_centers, tgt_colors, tgt_centers,
                             color_weight=color_weight, spatial_weight=proximity)

    N = cost.shape[0]
    print(f"Resolution: {resolution}x{resolution} -> N={N} cells. Building assignment...")

    if algorithm == 'optimal':
        print("Solving optimal assignment (Hungarian). This may be slow for large N.")
        rows, cols = assign_optimal(cost)
        # rows is 0..N-1 and cols[rows[i]] = assigned target
        src_to_tgt = np.zeros(N, dtype=int)
        src_to_tgt[rows] = cols
    else:
        print("Using greedy assignment (fast, approximate).")
        rows, src_to_tgt = assign_greedy(cost)

    out_img = render_output(srcc, tgtc, src_to_tgt, resolution)
    out_img.save(out_path)
    print(f"Saved output to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source image path (will be turned into Obama)")
    parser.add_argument("target", help="Target image path (Obama photo)")
    parser.add_argument("--out", "-o", default="out.png")
    parser.add_argument("--res", type=int, default=64, help="resolution per side (32,64 recommended)")
    parser.add_argument("--proximity", type=float, default=0.5, help="0..1 spatial weight")
    parser.add_argument("--color_weight", type=float, default=1.0)
    parser.add_argument("--algorithm", choices=['optimal','greedy'], default='optimal')
    args = parser.parse_args()

    obamify(args.source, args.target, args.out, resolution=args.res,
            proximity=args.proximity, color_weight=args.color_weight,
            algorithm=args.algorithm)
