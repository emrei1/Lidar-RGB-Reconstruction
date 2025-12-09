import pymeshlab as ml
import numpy as np

# Build a minimal mesh (1 triangle)
verts = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])

faces = np.array([[0, 1, 2]])

ms = ml.MeshSet()

# Use your version’s method: add_mesh(mesh)
mesh = ml.Mesh(vertex_matrix=verts, face_matrix=faces)
ms.add_mesh(mesh)

# Try saving as PLY
try:
    ms.save_current_mesh("test_output.ply")
    print("✔ PLY saving is supported")
except Exception as e:
    print("❌ PLY saving is NOT supported:", e)

