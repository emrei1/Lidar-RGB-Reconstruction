import argparse
import numpy as np
import open3d as o3d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    args = parser.parse_args()

    ply_path = f"{args.model_path}/point_cloud.ply"
    print("Loading:", ply_path)

    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError("PLY point cloud is empty or invalid")

    print("Loaded point cloud with", np.asarray(pcd.points).shape[0], "points")

    # Alpha Shape Meshing (fast)
    alpha = 0.05
    print("Meshing with alpha shape...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    mesh.compute_vertex_normals()

    out_path = f"{args.model_path}/reconstruction.obj"
    o3d.io.write_triangle_mesh(out_path, mesh)
    print("Saved OBJ:", out_path)


if __name__ == "__main__":
    main()

