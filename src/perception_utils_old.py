import numpy as np
from pydrake.all import (
    PointCloud,
    RigidTransform,
    RotationMatrix
)
from constants import BLOCK_COLOR_RGBA

def eval_point_cloud_from_system(diagram, context, pc_sys) -> PointCloud:
    pc_ctx = diagram.GetSubsystemContext(pc_sys, context)
    # try common port naming; fall back to port 0.
    try:
        port = pc_sys.GetOutputPort("point_cloud")
    except Exception:
        port = pc_sys.get_output_port(0)
    return port.Eval(pc_ctx)

def get_camera_clouds(diagram, context, pc_systems, cameras=("camera0", "camera1", "camera2")):
    clouds = []
    for cam in cameras:
        if isinstance(pc_systems, dict) and cam in pc_systems:
            clouds.append(eval_point_cloud_from_system(diagram, context, pc_systems[cam]))
        else:
            idx = int(cam.replace("camera", ""))
            clouds.append(eval_point_cloud_from_system(diagram, context, pc_systems[idx]))
    return clouds

def cloud_xyz_numpy(cloud: PointCloud) -> np.ndarray:
    """Return Nx3 float32 xyz from a Drake PointCloud."""
    if hasattr(cloud, "xyzs"):
        return np.asarray(cloud.xyzs()).T
    if hasattr(cloud, "mutable_xyzs"):
        return np.asarray(cloud.mutable_xyzs()).T
    # fallback: xyz(i) returns (3,1)
    N = cloud.size()
    out = np.empty((N, 3), dtype=np.float32)
    for i in range(N):
        out[i] = np.asarray(cloud.xyz(i)).reshape(3).astype(np.float32)
    return out

def pointcloud_from_xyz(xyz: np.ndarray) -> PointCloud:
    """
    xyz can be Nx3 or 3xN. Creates PointCloud and populates xyzs.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.size == 0:
        return PointCloud(0)

    if xyz.shape[0] == 3 and xyz.ndim == 2:
        xyz_3N = xyz
        N = xyz.shape[1]
    elif xyz.shape[1] == 3 and xyz.ndim == 2:
        N = xyz.shape[0]
        xyz_3N = xyz.T
    else:
        raise ValueError(f"Expected Nx3 or 3xN, got {xyz.shape}")

    pc = PointCloud(N)
    pc.mutable_xyzs()[:] = xyz_3N
    return pc

def crop_cloud_obb(
    cloud: PointCloud,
    X_WB: RigidTransform,
    half_extents: tuple[float, float, float] | list[float] | np.ndarray,
) -> PointCloud:
    """
    Crop a point cloud using an oriented bounding box aligned with the
    block frame B.

    Keeps only points p_W whose coordinates in the block frame p_B satisfy:
            |p_B.x| <= hx, |p_B.y| <= hy, |p_B.z| <= hz
    """
    # get Nx3 world points
    P_W = cloud_xyz_numpy(cloud).astype(float) # (N, 3)

    # pose of block B in world W
    R_WB = X_WB.rotation().matrix() # (3, 3)
    c_WB = X_WB.translation() # (3,)

    # transform points into block frame
    P_B = (R_WB.T @ (P_W - c_WB).T).T # (N, 3)

    hx, hy, hz = np.asarray(half_extents, dtype=float)

    # |x_B| <= hx, |y_B| <= hy, |z_B| <= hz
    mask = (
        (np.abs(P_B[:, 0]) <= hx) &
        (np.abs(P_B[:, 1]) <= hy) &
        (np.abs(P_B[:, 2]) <= hz)
    )

    # return new point cloud with only in-OBB points (in world coords)
    return pointcloud_from_xyz(P_W[mask])

def concat_clouds(clouds: list[PointCloud]) -> PointCloud:
    if not clouds:
        return PointCloud(0)
    xyz = np.vstack([cloud_xyz_numpy(c) for c in clouds])  # Nx3
    return pointcloud_from_xyz(xyz)

def remove_points_below_z(cloud: PointCloud, z_min: float = 0.01) -> PointCloud:
    xyz = cloud_xyz_numpy(cloud)
    mask = xyz[:, 2] >= z_min
    return pointcloud_from_xyz(xyz[mask])

def filter_cloud_by_color(
    cloud: PointCloud,
    color,
    eps: np.ndarray = np.array([100, 150, 150], dtype=np.uint8),
) -> PointCloud:
    """
    Given a Drake PointCloud with xyzs and rgbs, return a new PointCloud
    containing only the points whose RGB lies within [color - eps, color + eps].
    """
    if not hasattr(cloud, "rgbs"):
        return cloud

    rgbs = cloud.rgbs() # shape (3, N), dtype uint8
    xyzs = cloud.xyzs() # shape (3, N)

    # convert color to 3-channel uint8 in [0, 255]
    color_arr = np.asarray(color, dtype=float).reshape(-1)

    # if RGBA (4 channels), drop alpha
    if color_arr.size == 4:
        color_arr = color_arr[:3]

    assert color_arr.size == 3, f"Expected color with 3 or 4 components, got shape {color_arr.shape}"

    # handle either [0,1] floats or [0,255] floats/ints
    if color_arr.max() <= 1.0:
        # assume normalized [0,1]
        color_arr = (color_arr * 255.0)

    color_u8 = color_arr.astype(np.uint8).reshape(3, 1)
    # print("target_color: ", color_u8, " target color shape, ", color_u8.shape)
    eps = eps.reshape(3, 1)
    # print("eps: ", eps, " eps shape, ", eps.shape)

    color_i = color_u8.astype(np.int16)
    eps_i   = eps.astype(np.int16)
    lower_i = color_i - eps_i
    upper_i = color_i + eps_i

    lower_bound = np.clip(lower_i, 0, 255).astype(np.uint8)
    upper_bound = np.clip(upper_i, 0, 255).astype(np.uint8)

    # print(f"color lower bound: {lower_bound} [shape: {lower_bound.shape}], upper bound: {upper_bound}")
    # print("cloud_rgbs: ", rgbs.shape)

    # boolean mask over points (length N)
    mask = np.all((rgbs >= lower_bound) & (rgbs <= upper_bound), axis=0)

    print("color mask all failed: ", np.all(~mask))

    filtered_xyzs = xyzs[:, mask]  # shape (3, M)
    return pointcloud_from_xyz(filtered_xyzs)

def preprocess_block_cloud(diagram, context, X_WB_true, pc_systems,
                           half_extents=(0.18, 0.18, 0.18),
                           voxel=0.008, z_min=0.005):
    """
    Big perception processing system
    """
    # 1) get raw clouds from cameras
    clouds = get_camera_clouds(diagram, context, pc_systems)

    # 2) keep only specified-color block points in each cloud
    clouds = [
        filter_cloud_by_color(cl, BLOCK_COLOR_RGBA)
        for cl in clouds
    ]

    print("clouds ->", clouds)

    # 3) crop each cloud to the OBB box
    cropped = [crop_cloud_obb(cl, X_WB_true, half_extents) for cl in clouds]

    print(cropped, "<- cropped")

    # 4) merge and downsample
    merged = concat_clouds(cropped)
    merged = merged.VoxelizedDownSample(voxel_size=voxel)
    merged = remove_points_below_z(merged, z_min=z_min)

    return merged

def estimate_pose_pca(cloud: PointCloud) -> RigidTransform:
    if cloud.size() < 3:
        raise RuntimeError("Not enough points for PCA pose estimate")

    P = cloud_xyz_numpy(cloud).astype(float)
    c = P.mean(axis=0)
    Q = P - c

    # covariance (3x3)
    C = (Q.T @ Q) / max(Q.shape[0] - 1, 1)
    w, V = np.linalg.eigh(C)  # columns are eigenvectors
    V = V[:, np.argsort(w)[::-1]]  # sort by descending eigenvalue

    x, y, z = V[:, 0], V[:, 1], V[:, 2]

    # stabilize sign: prefer z pointing upward
    if np.dot(z, np.array([0.0, 0.0, 1.0])) < 0:
        z = -z

    # re-orthonormalize to guarantee a proper right-handed frame
    x = x / np.linalg.norm(x)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)

    R_WB = RotationMatrix(np.column_stack([x, y, z]))
    return RigidTransform(R_WB, c)


def estimate_extents_along_axes(cloud: PointCloud, X_WB_hat: RigidTransform):
    P = cloud_xyz_numpy(cloud).astype(float)
    R = X_WB_hat.rotation().matrix()
    c = X_WB_hat.translation()

    # transform points into the estimated local frame: p_B = R^T (p_W - c)
    local = (R.T @ (P - c).T).T  # Nx3

    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    return (maxs - mins), mins, maxs