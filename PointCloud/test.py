# https://www.matec-conferences.org/articles/matecconf/pdf/2016/38/matecconf_icmie2016_03005.pdf
# https://kr.mathworks.com/matlabcentral/fileexchange/55031-pointcloud2image-x-y-z-numr-numc
# https://www.semanticscholar.org/paper/The-Depth-Map-Construction-from-a-3D-Point-Cloud-Chmelar-Beran/d15c88bf744b159906bb780ba2f9cb6798cb15f1
#https://link.springer.com/article/10.1007/s11263-009-0297-y

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

pcd = o3d.io.read_point_cloud("20220719_174345_17.pcd")
# pcdnp = np.asarray(pcd.points)
vis= o3d.visualization.Visualizer()

vis.create_window('pcl',640, 480, 50, 50, True)
vis.add_geometry(pcd)

vis = o3d.visualization.Visualizer()
vis.create_window(visible = True)
vis.add_geometry(pcd)

depth = vis.capture_depth_float_buffer(False)
image = vis.capture_screen_float_buffer(False)
plt.imshow(np.asarray(depth))

o3d.io.write_image("./test_depth.png", depth)

plt.imsave("./test_depth.png", np.asarray(depth))#, dpi = 1)

from PIL import Image
img = Image.open('./test_depth.png').convert('LA')
img.save('./greyscale.png')