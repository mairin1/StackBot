import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from pydrake.all import(
    RigidTransform,
    RotationMatrix
)

from constants import *

def block_directive(block_name : str, translation, rotation) -> str:
    assert len(translation) == 3
    assert len(rotation) == 3

    x, y, z = translation
    r, p, yaw = rotation

    return f"""
    - add_model:
        name: {block_name}
        file: package://stackbot/{block_name}.sdf
        default_free_body_pose:
            {block_name}_link:
                translation: [{x}, {y}, {z}]
                rotation: !Rpy {{ deg: [{r}, {p}, {yaw}]}}
"""

def create_randomized_block_directives(block_names : list[str]) -> str:
    directives = ""

    def sample_point():
        """
        sample a point on the xy plane such that |x|, |y| > inner_d, |x|, |y| < outer_d
        (think concentric squares)
        """
        inner_d = 0.2
        outer_d = 0.8

        x, y = np.random.uniform(low=inner_d, high=outer_d, size=2) * np.random.choice([-1, 1], size=2)
        return x, y

    def sample_points(n : int, reject_dist = 0.3, num_retries = 100):
        d_squared = reject_dist ** 2

        for _ in range(num_retries):
            points = []
            for _ in range(n):
                x, y = sample_point()
                if any(map(lambda p : (p[0] - x)**2 + (p[1] - y) ** 2 < d_squared, points)):
                    break
                points.append((x, y))
            
            if len(points) == n:
                return points
        
        raise Exception("Exceeded max retries")
           
    coords = sample_points(len(block_names))

    for block_name, coord in zip(block_names, coords):
        x, y = coord
        z = 0.05
        rotation = np.random.randint(0, 360)

        directives += block_directive(block_name, [x, y, z], [0, 0, rotation])

    return directives

def create_block_directives_from_file(filename: str) -> str:
    """
    load a json file for block positions
    """
    with open(filename, "r") as f:
        positions = json.load(f)

    directives = ""
    
    for body_name, pose in positions.items():
        if "block" in body_name:
            block_name = body_name.split("_")[0] # hacky, block10_link -> block10
            directives += block_directive(block_name, pose["translation"], np.array(pose["rotation"]) * 180 / np.pi)
    
    return directives

def create_camera_directives() -> str:
    return """
    - add_frame:
        name: camera0_origin
        X_PF:
            base_frame: world
            rotation: !Rpy { deg: [-120.0, 0.0, 180.0]}
            translation: [0, 1.0, 0.8]

    - add_model:
        name: camera0
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera0_origin
        child: camera0::base

    - add_frame:
        name: camera1_origin
        X_PF:
            base_frame: world
            rotation: !Rpy { deg: [-125, 0.0, 90.0]}
            translation: [1.0, 0.1, 0.8]

    - add_model:
        name: camera1
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera1_origin
        child: camera1::base

    - add_frame:
        name: camera2_origin
        X_PF:
            base_frame: world
            rotation: !Rpy { deg: [-120.0, 0.0, -90.0]}
            translation: [-1.0, 0.1, 0.8]

    - add_model:
        name: camera2
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera2_origin
        child: camera2::base
"""

def generate_scenario_yaml(blocks: list[str]) -> str:
    directives = f"""
directives:
    - add_model:
        name: iiwa
        file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
        default_joint_positions:
            iiwa_joint_1: [-1.57]
            iiwa_joint_2: [0.1]
            iiwa_joint_3: [0]
            iiwa_joint_4: [-1.2]
            iiwa_joint_5: [0]
            iiwa_joint_6: [ 1.6]
            iiwa_joint_7: [0]
    - add_weld:
        parent: world
        child: iiwa::iiwa_link_0
        X_PC:
            translation: [0, -0.5, 0]
            rotation: !Rpy {{ deg: [0, 0, 180] }}
    - add_model:
        name: wsg
        file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
    - add_weld:
        parent: iiwa::iiwa_link_7
        child: wsg::body
        X_PC:
            translation: [0, 0, 0.09]
            rotation: !Rpy {{deg: [90, 0, 90]}}
    - add_model:
        name: floor
        file: package://stackbot/floor.sdf
    - add_weld:
        parent: world
        child: floor::floor_link
        X_PC:
            translation: [0, 0, -0.05]
    - add_model:
        name: platform
        file: package://stackbot/platform.sdf
    - add_weld:
        parent: world
        child: platform::platform_link

    { create_randomized_block_directives(blocks) }
    { create_camera_directives() }
    """

    cameras = """
cameras:
    camera0:
        name: camera0
        depth: True
        X_PB:
            base_frame: camera0::base
    camera1:
        name: camera1
        depth: True
        X_PB:
            base_frame: camera1::base
    camera2:
        name: camera2
        depth: True
        X_PB:
            base_frame: camera2::base
"""

    drivers = """
model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
"""

    return directives + cameras + drivers

# def randomize_blocks_near_kuka(
#     block_names,
#     plant,
#     context,
#     base_model_name: str = "iiwa",
#     base_link_name: str = "iiwa_link_0",
#     min_radius: float = 0.35,
#     max_radius: float = 0.75,
#     fov_degrees: float = 100.0,  # cone around direction to platform
#     platform_model_name: str = "platform",
#     platform_link_name: str = "platform_link",
#     platform_exclusion_radius: float = 0.30,
#     max_tries_per_block: int = 100,
#     rng: np.random.Generator | None = None,
# ) -> None:
#     """
#     Randomizes blocks in an annulus [min_radius, max_radius] around the iiwa
#     base, but only in a cone that points from the base toward the platform.
#     Also avoids a disk around the platform (where we'll stack).

#     Geometry is all in XY; Z and rotation are preserved.
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     base_instance = plant.GetModelInstanceByName(base_model_name)
#     base_body = plant.GetBodyByName(base_link_name, base_instance)
#     X_WBase = plant.EvalBodyPoseInWorld(context, base_body)
#     p_WBase = X_WBase.translation()

#     platform_instance = plant.GetModelInstanceByName(platform_model_name)
#     platform_body = plant.GetBodyByName(platform_link_name, platform_instance)
#     X_WPlat = plant.EvalBodyPoseInWorld(context, platform_body)
#     p_WPlat = X_WPlat.translation()

#     # platform footprint (known from how we created it)
#     plat_size = np.array([0.5, 0.5, 0.05])
#     plat_half = plat_size[:2] / 2.0

#     # conservative block footprint (max of generated blocks)
#     # w <= 0.10, l <= 0.30
#     block_half = np.array([0.10, 0.30]) / 2.0

#     # exclusion half-extent = platform half + block half + small margin
#     margin = 0.02
#     exclude_half = plat_half + block_half + margin

#     # front = direction from base to platform in XY
#     front_xy = p_WPlat[:2] - p_WBase[:2]
#     if np.linalg.norm(front_xy) < 1e-6:
#         front_xy = np.array([0.0, 1.0])  # fallback
#     front_xy = front_xy / np.linalg.norm(front_xy)
#     cos_fov_half = np.cos(np.deg2rad(fov_degrees) / 2.0)

#     for name in block_names:
#         instance = plant.GetModelInstanceByName(name)
#         body_indices = list(plant.GetBodyIndices(instance))
#         if not body_indices:
#             raise RuntimeError(f"No bodies found for block model '{name}'")

#         block_body = plant.get_body(body_indices[0])
#         z = BLOCK_CENTER_Z

#         for _ in range(max_tries_per_block):
#             # sample radius
#             r = np.sqrt(
#                 (max_radius**2 - min_radius**2) * rng.random()
#                 + min_radius**2
#             )
#             theta = 2.0 * np.pi * rng.random()
#             dx, dy = r * np.cos(theta), r * np.sin(theta)
#             candidate_xy = np.array([p_WBase[0] + dx, p_WBase[1] + dy])

#             # must lie in cone pointing to platform
#             v = candidate_xy - p_WBase[:2]
#             if np.linalg.norm(v) < 1e-6:
#                 continue
#             v = v / np.linalg.norm(v)
#             cos_angle = np.dot(v, front_xy)
#             if cos_angle < cos_fov_half:
#                 continue  # too far to the side / behind

#             # must be outside platform exclusion radius
#             if np.linalg.norm(candidate_xy - p_WPlat[:2]) < platform_exclusion_radius:
#                 continue

#             delta = candidate_xy - p_WPlat[:2]
#             if (np.abs(delta) <= exclude_half).all():
#                 # inside forbidden rectangle -> would intersect or be under platform
#                 continue

#             # accept this sample :)
#             p_WB_new = np.array([candidate_xy[0], candidate_xy[1], z])
#             yaw = rng.uniform(0.0, 2.0 * np.pi)
#             R_WB_new = RotationMatrix.MakeZRotation(yaw)
#             X_WB_new = RigidTransform(R_WB_new, p_WB_new) # type: ignore
#             plant.SetFreeBodyPose(context, block_body, X_WB_new)
#             break
#         else:
#             print(
#                 f"[randomize_blocks_near_kuka] Warning: "
#                 f"could not find valid pose for {name} after {max_tries_per_block} tries."
#             )