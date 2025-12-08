import numpy as np
import typing
from pydrake.all import (
    Concatenate,
    Context,
    DiagramBuilder,
    InverseKinematics,
    MultibodyPlant,
    PiecewisePolynomial,
    PointCloud,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
    StartMeshcat,
    TrajectorySource,
    ConstantVectorSource
)
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

block_color = np.array([0, 255, 0])


"""
Given a point cloud with rgbs and xyzs, returns a new point cloud of
all the points within the offset of the given rgb color.
"""
def isolate_blocks_by_color(point_cloud: PointCloud, color: np.array, eps: np.array = [100, 150, 150]):

    xyzs = point_cloud.xyzs()
    rgbs = point_cloud.rgbs()

    lower_bound = (color - eps).reshape(3, 1)
    upper_bound = (color + eps).reshape(3, 1)

    mask = np.all((rgbs >= lower_bound) & (rgbs <= upper_bound), axis=0)
    filtered_xyzs = xyzs[:, mask]

    new_pc = PointCloud(new_size=filtered_xyzs.shape[1])
    new_pc.mutable_xyzs()[:] = filtered_xyzs
    print(filtered_xyzs.shape)

    return new_pc

def dbscan(eps: float, min_samples: int, points: np.array):
    points = points[:].T
    assert (np.shape(points)[1] == 3)

    scaler = MinMaxScaler()
    scaled_pts = scaler.fit_transform(points)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_pts)
    return clustering

def clusters_to_point_clouds(dbscan_obj, points, meshcat, display=True):
    _, n_features = dbscan_obj.components_.shape
    labels = dbscan_obj.labels_
    pcs = []

    for n in range(n_features+1):
        mask = np.array(list(map(lambda x: x == n, labels)))
        points_in_cluster_n = points.T[mask]
        
        pc = PointCloud(new_size=points_in_cluster_n.shape[0])
        pc.mutable_xyzs()[:] = points_in_cluster_n.T
        pcs.append(pcs)

        if display:
            color = Rgba(np.random.rand(), np.random.rand(), np.random.rand())
            meshcat.SetObject(
            f"cluster_{n}", pc, point_size=0.05, rgba=color
            )

    return pcs

# calculate approximate midpoint of the block
def calc_translation(pointCloud):
    points = pointCloud.xyzs()
    xyz = np.zeros(3)
    for i in range(3):
        xyz[i] = np.average(points[i])
    return xyz

def calculate_all_translations(block_pcs, meshcat, display=True):
    translations = np.zeros((len(block_pcs), 3))
    for i, block_pc in enumerate(block_pcs):
        translation = calc_translation(block_pc)
        translations[i] = translation
    if display:
        display_pc = PointCloud(len(translations))
        display_pc.mutable_xyzs()[:] = translations.T
        # TODO: it should be a sphere but i will fix that later :)))
        meshcat.SetObject("midpoints", display_pc, point_size=0.5, rgba=Rgba(0, 1, 1, 1))
    return translations

def euclidean_dist(xyz1, xyz2):
    diffs = [(a - b) ** 2 for a, b in zip(xyz1, xyz2)]
    return sum(diffs) ** 0.5

def estimate_block_length(block_pc):
    points = block_pc.xyzs()
    xs, ys = points[0], points[1]
    corner1 = points[:, np.argmin(xs)]
    corner2 = points[:, np.argmax(xs)]
    corner3 = points[:, np.argmin(ys)]
    corner4 = points[:, np.argmax(ys)]
    # ignore z values for now
    sideA = euclidean_dist(corner1[:2], corner2[:2])
    sideB = euclidean_dist(corner3[:2], corner4[:2])
    # TODO: what is best metric for this?
    # ideas: max to deal with incomplete point clouds
    # average
    return max(sideA, sideB)

def estimate_block_lengths(block_pt_clouds):
    lengths = np.zeros((len(block_pt_clouds), 1))
    for i, block_pc in enumerate(block_pt_clouds):
        length = estimate_block_length(block_pc)
        lengths[i] = length
    return lengths

def lengths_to_i_vals(lengths):
    return np.round((lengths - 0.1)/0.02)

def slope(xy1, xy2):
    if (xy2[0] == xy1[0]):
        return float("inf")
    dy = xy2[1] - xy1[1]
    dx = xy2[0] - xy1[0]
    return np.array([dy/dx, dy, dx])

def calc_rotation(block_pc):
    points = block_pc.xyzs()
    xs, ys = points[0], points[1]
    corner1 = points[:, np.argmin(xs)]
    corner2 = points[:, np.argmax(xs)]
    corner3 = points[:, np.argmin(ys)]
    corner4 = points[:, np.argmax(ys)]
    slopes = np.array([slope(corner1[:2], corner4[:2]),
              slope(corner2[:2], corner3[:2])])
    best_slope_ind = np.argmax(slopes, axis=0)[0]
    dy, dx = slopes[best_slope_ind][1], slopes[best_slope_ind][2]
    angle_from_z = np.pi/2 - np.arctan2(dy, dx)
    return angle_from_z
