# down sampling
import open3d as o3d
import pptk

pcd = o3d.io.read_point_cloud("pcd/20220719_174345_17.pcd")
print(f"Points before downsampling: {len(pcd.points)} ")
# Points before downsampling: 87141
pcd = pcd.voxel_down_sample(voxel_size=0.2)
print(f"Points after downsampling: {len(pcd.points)}")
# Points after downsampling: 19900
v = pptk.viewer(pcd.points)