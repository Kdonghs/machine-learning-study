# 도로와 객체를 분리
# inlier와 outlier를 구분
# red dot = inlier, gray dot = outlier
import open3d as o3d
import numpy as np
import pptk
import time
import matplotlib as plt
import hdbscan

pcd = o3d.io.read_point_cloud("pcd/20220719_174345_17.pcd")
pcd_np = np.asarray(pcd.points)

t1 = time.time()
plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=100)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
outlier_cloud.paint_uniform_color([1, 0, 0])
t2 = time.time()
print(f"Time to segment points using RANSAC {t2 - t1}")
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# CLUSTERING WITH DBSCAN
t3 = time.time()
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.60, min_points=50, print_progress=False))

max_label = labels.max()
print(f'point cloud has {max_label + 1} clusters')
colors = plt.cm.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
t4 = time.time()
print(f'Time to cluster outliers using DBSCAN {t4 - t3}')
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# CLUSTERING WITH HDBSCAN
t3 = time.time()
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=True)
clusterer.fit(np.array(outlier_cloud.points))
labels = clusterer.labels_

max_label = labels.max()
print(f'point cloud has {max_label + 1} clusters')
colors = plt.cm.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
t4 = time.time()
print(f'Time to cluster outliers using HDBSCAN {t4 - t3}')
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])