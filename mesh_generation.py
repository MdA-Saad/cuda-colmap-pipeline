import open3d as o3d
import numpy as np
import os

# --- Configuration ---
input_file = "data/workspace/mvs/fuse.ply"
output_file = "data/workspace/mvs/mesh_south_building.ply"

print("--- Starting Open3D Meshing Pipeline ---")

if not os.path.exists(input_file):
    print(f"ERROR: Cannot find '{input_file}' in the current directory.")
    print(f"Current working directory is: {os.getcwd()}")
    exit()

pcd = o3d.io.read_point_cloud(input_file)

# Remove noise and floaters
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Estimate Normals
print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)

# depth=8 keeps RAM usage manageable during calculation
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.1, linear_fit=True)

# Density trimming
densities = np.asarray(densities)
density_threshold = np.quantile(densities, 0.05)
vertices_to_remove = densities < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)

# Taubin smoothing: Iron out the melted SfM noise without shrinking
mesh = mesh.filter_smooth_taubin(number_of_iterations=15)

print(f"    -> Raw mesh generated with {len(mesh.triangles)} triangles.")

# Export binary PLY
generated_mesh = o3d.io.write_triangle_mesh(output_file, mesh, write_ascii=False)

if not generated_mesh:
    print("ERROR: Open3D generated the mesh but failed to write the file to the disk.")
