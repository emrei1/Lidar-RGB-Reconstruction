import open3d as o3d
import numpy as np
import argparse
import os

def load_gaussians_from_ply(ply_path):
    print(f"Loading PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    if pcd.is_empty():
        raise RuntimeError("PLY file is empty or invalid")

    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals) if len(pcd.normals) else None

    print("Loaded:")
    print(" - xyz:", xyz.shape)
    print(" - colors:", colors.shape)
    if normals is not None:
        print(" - normals:", normals.shape)

    return xyz, colors, normals


def save_as_obj(obj_path, xyz, colors):
    print(f"Saving OBJ to: {obj_path}")

    with open(obj_path, "w") as f:
        for (x, y, z), (r, g, b) in zip(xyz, colors):
            f.write(f"v {x} {y} {z} {r} {g} {b}\n")

    print("OBJ saved successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True, help="Path to model folder")
    args = parser.parse_args()

    model_path = args.model_path.rstrip("/")
    ply_candidate_dir = os.path.join(model_path, "point_cloud")

    # autodetect iteration folder
    if not os.path.exists(ply_candidate_dir):
        raise RuntimeError("No point_cloud folder found in model path")

    # find the latest iteration folder
    sub = sorted(os.listdir(ply_candidate_dir))
    if len(sub) == 0:
        raise RuntimeError("No iteration folders inside point_cloud")

    latest_iter = sub[-1]  # last folder

    ply_path = os.path.join(ply_candidate_dir, latest_iter, "point_cloud.ply")

    if not os.path.exists(ply_path):
        raise RuntimeError(f"PLY file not found at: {ply_path}")

    xyz, colors, normals = load_gaussians_from_ply(ply_path)

    # Output OBJ
    obj_path = os.path.join(model_path, "reconstruction.obj")
    save_as_obj(obj_path, xyz, colors)

    print("\nDone! OBJ saved at:")
    print(obj_path)


if __name__ == "__main__":
    main()

