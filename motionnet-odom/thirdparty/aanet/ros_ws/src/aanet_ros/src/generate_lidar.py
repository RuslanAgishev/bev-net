import numpy as np
# import cupy as cp


def project_disp_to_points(calib, disp, max_high):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid], valid

def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

from sensor_msgs.msg import PointCloud2, PointField
def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    # msg.data = points_sum.astype(np.float32).tobytes()
    return msg

def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id='camera_link', seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.

    points.shape = (N, 3)
    colors.shape = (N,3)
    '''
    colors = denormalize(colors)
    msg = PointCloud2()

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1),
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True;
    msg.data = xyzrgb.tostring()
    return msg


def pto_rec_map(velo_points, H=64, W=512, D=800):
    # depth, width, height
    valid_inds = (velo_points[:, 0] < 80) & \
                 (velo_points[:, 0] >= 0) & \
                 (velo_points[:, 1] < 50) & \
                 (velo_points[:, 1] >= -50) & \
                 (velo_points[:, 2] < 1) & \
                 (velo_points[:, 2] >= -2.5)
    velo_points = velo_points[valid_inds]

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
    x_grid = (x * D / 80.).astype(int)
    x_grid[x_grid < 0] = 0
    x_grid[x_grid >= D] = D - 1

    y_grid = ((y + 50) * W / 100.).astype(int)
    y_grid[y_grid < 0] = 0
    y_grid[y_grid >= W] = W - 1

    z_grid = ((z + 2.5) * H / 3.5).astype(int)
    z_grid[z_grid < 0] = 0
    z_grid[z_grid >= H] = H - 1

    depth_map = - np.ones((D, W, H, 4))
    depth_map[x_grid, y_grid, z_grid, 0] = x
    depth_map[x_grid, y_grid, z_grid, 1] = y
    depth_map[x_grid, y_grid, z_grid, 2] = z
    depth_map[x_grid, y_grid, z_grid, 3] = i
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def pto_ang_map(velo_points, color_points=None, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    if color_points is not None:
        color_map = - np.ones((H, W, 4))
        color_map[theta_, phi_, 0] = color_points[:, 0]
        color_map[theta_, phi_, 1] = color_points[:, 1]
        color_map[theta_, phi_, 2] = color_points[:, 2]
        color_map[theta_, phi_, 3] = color_points[:, 3]
        color_map = color_map[0::slice, :, :]
        color_map = color_map.reshape((-1, 4))
        color_map = color_map[color_map[:, 0] != -1.0]
        return depth_map, color_map
    else:
        return depth_map, None


def gen_sparse_points(pc_velo, color_points=None, H=64, W=512, slice=1):
    """
    :param pc_velo: pc_velo.shape = (-1, 4)
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    :return:
    """
    pc_velo = np.concatenate([pc_velo, np.ones((pc_velo.shape[0], 1))], 1) # pad 1 in the intensity dimension
    # depth, width, height
    valid_inds = (pc_velo[:, 0] < 120) & \
                 (pc_velo[:, 0] >= 0) & \
                 (pc_velo[:, 1] < 50) & \
                 (pc_velo[:, 1] >= -50) & \
                 (pc_velo[:, 2] < 1.5) & \
                 (pc_velo[:, 2] >= -2.5)
    pc_velo = pc_velo[valid_inds]
    if color_points is not None:
        color_points = np.concatenate([color_points, np.ones((color_points.shape[0], 1))], 1) # pad 1 in the intensity dimension
        color_points = color_points[valid_inds]

    return pto_ang_map(pc_velo, color_points, H=H, W=W, slice=slice)
