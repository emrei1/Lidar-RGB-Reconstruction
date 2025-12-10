import torch
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_depth_from_transi(gt_transi, output_dir="test/transient_depth"):
    """
    Visualize depth encoded in gt_transi histogram.
    gt_transi: (H, W, Bins)
    """

    ensure_dir(output_dir)

    if gt_transi.ndim != 3:
        raise ValueError(f"gt_transi must be (H, W, Bins). Got shape {gt_transi.shape}")

    H, W, B = gt_transi.shape
    gt = gt_transi.float()

    # =========================
    # 1. ARGMAX DEPTH (H,W)
    # =========================
    argmax_depth = torch.argmax(gt, dim=2).float()   # (H, W)
    argmax_depth_norm = argmax_depth / (argmax_depth.max() + 1e-8)

    save_image(argmax_depth_norm.unsqueeze(0),
               f"{output_dir}/depth_argmax.png")
    print(f"[Saved] Argmax depth → {output_dir}/depth_argmax.png")

    # =========================
    # 2. EXPECTED DEPTH (weighted average over bins)
    # =========================
    bins = torch.arange(B, dtype=torch.float32, device=gt.device).view(1, 1, B)
    weights = gt / (gt.sum(dim=2, keepdim=True) + 1e-8)

    expected_depth = (weights * bins).sum(dim=2)  # (H, W)
    expected_depth_norm = expected_depth / (expected_depth.max() + 1e-8)

    save_image(expected_depth_norm.unsqueeze(0),
               f"{output_dir}/depth_expected.png")
    print(f"[Saved] Expected depth → {output_dir}/depth_expected.png")

    # =========================
    # 3. COLORIZED DEPTH (using matplotlib)
    # =========================
    depth_np = expected_depth_norm.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(depth_np, cmap="turbo")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/depth_color.png", dpi=200)
    plt.close()

    print(f"[Saved] Colorized depth → {output_dir}/depth_color.png")


gt_transi = np.load("MyUnityScene/transient/frame_0005_transient.npy")

# Rotate 90 degrees counter-clockwise and FIX negative strides using .copy()
gt_transi_rot = np.rot90(gt_transi, k=1, axes=(0, 1)).copy()

# Convert to torch
gt_transi_rot = torch.from_numpy(gt_transi_rot).float()

# Visualize
visualize_depth_from_transi(gt_transi_rot)
