import cv2
import numpy as np
import pandas as pd
import matplotlib as plt
import open3d as o3d

class LiDAR2CameraKITTI(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)

        P = calibs["P2"]
        self.P = np.reshape(P, [3, 4])

        # Rigid transform from Velodyne coord to reference camera coord
        V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(V2C, [3, 4])

        # Rotation from reference camera coord to rect camera coord
        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, [3, 3])

        self.img = None
        self.point_cloud = None
        self.cam_point_cloud = None

        self.imgfov_points_2d = None
        self.imgfov_cam_point_cloud = None
        self.imgfov_depth = None
        self.imgfov_depthmap = None

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def get_image(self, image_path):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return img

    def get_point_cloud(self, point_cloud_path):
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape((-1, 4))
        point_cloud = point_cloud[:, :3]
        return point_cloud

    def get_cam_point_cloud(self, point_cloud):
        # point_cloud : (n, 3) → point_cloud_homo : (n, 4)
        point_cloud_homo = np.column_stack([point_cloud, np.ones((point_cloud.shape[0], 1))])
        # lidar to cam X point_cloud_homo.T : (3, 4) x (4, n) = (3, n)
        cam_point_cloud = np.dot(self.V2C, np.transpose(point_cloud_homo))
        # cam_point_cloud.T : (3, n) → (n, 3)
        cam_point_cloud = cam_point_cloud.T
        return cam_point_cloud

    def project_point_cloud_to_image(self, cam_point_cloud, debug=False):
        '''
        Input: 3D points in Velodyne Frame [nx3]
        Output: 2D Pixels in Image Frame [nx2]
        '''

        # R0 : (3, 3) → R0_homo : (4, 4)
        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo = np.column_stack([R0_homo, [0, 0, 0, 1]])

        # P x R0 : (3, 4) x (4, 4)
        p_r0 = np.dot(self.P, R0_homo)

        # point_cloud in camera : (n, 3) → point_cloud_homo in camera : (n, 4)
        cam_point_cloud_homo = np.column_stack([cam_point_cloud, np.ones((cam_point_cloud.shape[0], 1))])

        # P x RO x X : (3, 4) x (4, 4) x (4, n) → (3, n)
        p_r0_x = np.dot(p_r0, np.transpose(cam_point_cloud_homo))

        # points_2d : (n, 3)
        points_2d = np.transpose(p_r0_x)

        if debug == True:
            print("R0_homo : \n", R0_homo)
            print("")
            print("p_r0 : \n", p_r0)
            print("")
            print("cam_point_cloud_homo : \n", cam_point_cloud_homo)
            print("")
            print("p_r0_x : \n", p_r0_x)
            print("")
            print("points_2d : \n", points_2d)
            print("")

        # points_2d : cartesian coodrdinate (u, v)
        points_2d[:, 0] /= points_2d[:, 2]
        points_2d[:, 1] /= points_2d[:, 2]

        if debug == True:
            print("points_2d in cartesian : \n", points_2d[:, 0:2])
            print("")

        return points_2d[:, 0:2]

    def get_points_in_image_fov(self, cam_point_cloud, xmin, ymin, xmax, ymax, clip_distance=0, debug=False):
        """ Filter point cloud, keep those in image FOV """

        # point cloud in camera → points in 2d image : (n, 2)
        points_2d = self.project_point_cloud_to_image(cam_point_cloud, debug)
        points_2d = np.round(points_2d)

        # points index in fov
        fov_inds = (
                (points_2d[:, 0] < xmax)
                & (points_2d[:, 0] >= xmin)
                & (points_2d[:, 1] < ymax)
                & (points_2d[:, 1] >= ymin)
        )

        # depth orientation in camera coordinate is 2
        fov_inds = fov_inds & (cam_point_cloud[:, 2] > clip_distance)

        # imgfov_cam_point_cloud : (K, 3)
        imgfov_cam_point_cloud = cam_point_cloud[fov_inds, :]
        # points_2d : (K, 2)
        imgfov_points_2d = points_2d[fov_inds, :]

        return imgfov_cam_point_cloud, imgfov_points_2d

    def get_min_dist_points_in_image_fov(self, imgfov_cam_point_cloud, imgfov_points_2d):

        df = pd.DataFrame({
            'width': imgfov_points_2d[:, 0],
            'height': imgfov_points_2d[:, 1],
            'X': imgfov_cam_point_cloud[:, 0],
            'Y': imgfov_cam_point_cloud[:, 1],
            'Z': imgfov_cam_point_cloud[:, 2]
        })

        # Z is depth in camera coordiante
        min_depth_df = df.groupby(['width', 'height', 'X', 'Y'], as_index=False).min()

        min_depth_np = np.array(min_depth_df)
        imgfov_points_2d = np.c_[min_depth_df['width'].to_numpy(), min_depth_df['height'].to_numpy()]
        imgfov_cam_point_cloud = np.c_[
            min_depth_df['X'].to_numpy(), min_depth_df['Y'].to_numpy(), min_depth_df['Z'].to_numpy()]

        return imgfov_cam_point_cloud, imgfov_points_2d

    def get_projected_image(self, image_path, point_cloud_path, range_meter=100.0, min_depth_filter=True, debug=False):
        """ Project LiDAR points to image """

        # origin img and point cloud
        self.img = self.get_image(image_path)
        # point_cloud in lidar: (n, 3)
        self.point_cloud = self.get_point_cloud(point_cloud_path)
        # point_cloud in camera: (n, 3)
        self.cam_point_cloud = self.get_cam_point_cloud(self.point_cloud)

        # imgfov_point_cloud : (K, 3), imgfov_points_2d : (K, 2)
        imgfov_cam_point_cloud, imgfov_points_2d = self.get_points_in_image_fov(
            self.cam_point_cloud, 0, 0, self.img.shape[1], self.img.shape[0], debug=debug)

        if min_depth_filter:
            imgfov_cam_point_cloud, imgfov_points_2d = self.get_min_dist_points_in_image_fov(
                imgfov_cam_point_cloud, imgfov_points_2d)

        # imgfov_points_2d : (N, 2) with (u, v) coordinate
        self.imgfov_points_2d = imgfov_points_2d
        # imgfov_point_cloud : (N, 3) with (X, Y, Z) coordinate
        self.imgfov_cam_point_cloud = imgfov_cam_point_cloud
        # imgfov_depth : (N, 1)
        self.imgfov_depth = imgfov_cam_point_cloud[:, 2]
        # imgfov_depthmap : (H, W)
        self.imgfov_depthmap = np.zeros((self.img.shape[:2]))
        row_idx, col_idx = self.imgfov_points_2d[:, 1].astype(np.int16), self.imgfov_points_2d[:, 0].astype(np.int16)
        self.imgfov_depthmap[row_idx, col_idx] = self.imgfov_depth

        cmap = plt.cm.get_cmap("jet", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        img = self.img.copy()
        for i in range(self.imgfov_points_2d.shape[0]):
            depth = self.imgfov_depth[i]
            # set color in range from 0 to range_meter (ex. 50 m)
            color_index = int(255 * min(depth, range_meter) / range_meter)
            color = cmap[color_index, :]
            cv2.circle(
                img, (int(np.round(self.imgfov_points_2d[i, 0])), int(np.round(self.imgfov_points_2d[i, 1]))), 2,
                color=tuple(color),
                thickness=-1)

        return img
pcd = o3d.io.read_point_cloud("20220719_174345_17.pcd")
pcdnp = np.asarray(pcd.points)
LiDAR2CameraKITTI.project_point_cloud_to_image()