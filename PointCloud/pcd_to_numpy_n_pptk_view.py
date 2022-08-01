import open3d as o3d
import numpy as np


# point cloud → numpy
import pptk

pcd = o3d.io.read_point_cloud("pcd/20220719_174345_17.pcd")
pcd_np = np.asarray(pcd.points)
print(pcd_np)
# array([[896.994  ,  48.7601 ,  82.2656 ],
#        [906.593  ,  48.7601 ,  80.7452 ],
#        [907.539  ,  55.4902 ,  83.6581 ],
#        ...,
#        [806.665  , 627.363  ,   5.11482],
#        [806.665  , 654.432  ,   7.51998],
#        [806.665  , 681.537  ,   9.48744]])

# numpy → point cloud
A = np.random.random((1000, 3)) * 1000
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(A)
# open3d.visualization.draw_geometries([pcd])
v = pptk.viewer(pcd.points)
# v = pptk.viewer(np.asarray(pcd.points))