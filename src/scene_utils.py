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

def create_block_sdf(
        model_name : str, 
        size : np.ndarray, 
        mass : float = 1, 
        pose : np.ndarray = np.array([0, 0, 0, 0, 0, 0]),
        rgba : np.ndarray = np.array([1, 1, 1, 1]), 
        mu_static: float = 2.0,
        mu_dynamic: float = 1.5,
) -> str:
    assert len(pose) == 6, "pose must be 6d"
    assert len(size) == 3, "size must be 3d"

    a, b, c = size
    ixx = mass * (b**2 + c**2) / 12
    iyy = mass * (a**2 + c**2) / 12
    izz = mass * (a**2 + b**2) / 12

    return (
f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">
    <pose> {" ".join(str(num) for num in pose) } </pose>
    <link name="{model_name}_link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{iyy}</iyy>
          <iyz>0.0</iyz>
          <izz>{izz}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size> {" ".join(str(num) for num in size)} </size>
          </box>
        </geometry>
        <surface>
            <friction>
                <ode>
                    <mu>{mu_static}</mu>
                    <mu2>{mu_dynamic}</mu2>
                </ode>
            </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size> {" ".join(str(num) for num in size)} </size>
          </box>
        </geometry>
        <material>
          <diffuse>{" ".join(str(num) for num in rgba)}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")

def _export_box_obj(path: str | Path, size_xyz):
    """Writes an axis-aligned box mesh centered at the origin with extents=size_xyz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=np.asarray(size_xyz, dtype=float))
    mesh.export(path.as_posix())  # .obj inferred from extension

def generate_blocks():
    assets_dir = Path("assets/models")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # I want blocks to be 10 x (10 + 2i) x 8 cm
    for i in range(11):
        w = min(0.06 + 0.01 * i, 0.1) # i think 10cm is about our max gripper width
        l = 0.1 + 0.02 * i # 10 + i cm   
        h = BLOCK_HEIGHT
        size = np.array([w, l, h])
        with open(f"assets/models/block{i}.sdf", "w+") as f:
            f.write(create_block_sdf(f"block{i}", size, rgba=BLOCK_COLOR_RGBA, mu_static=20, mu_dynamic=15))
        _export_box_obj(assets_dir / f"block{i}.obj", size)

    # I want a floor
    with open("assets/models/floor.sdf", "w+") as f:
        f.write(create_block_sdf("floor", np.array([3, 3, 0.1]), rgba=np.array([0.1, 0.1, 0.1, 1])))

    # I want a platform to stack on
    with open("assets/models/platform.sdf", "w+") as f:
        f.write(create_block_sdf("platform", np.array([0.5, 0.5, 0.05])))

def create_camera_directives() -> None:
    camera_directives_yaml = """directives:
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: [-120.0, 0.0, 180.0]}
        translation: [0, 0.8, 0.5]

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
        translation: [0.8, 0.1, 0.5]

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
        translation: [-0.8, 0.1, 0.5]

- add_model:
    name: camera2
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera2_origin
    child: camera2::base
"""
    os.makedirs("directives", exist_ok=True)
    with open("directives/camera_directives.dmd.yaml", "w") as f:
        f.write(camera_directives_yaml)

def generate_directives_yaml(num_blocks: int):
    directives = f"""directives:
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
    """

    blocks = [f"block{i}" for i in range(1, num_blocks + 1)]

    block_sdfs = [f"""
- add_model:
    name: {block}
    file: package://stackbot/{block}.sdf
    """ for block in blocks]
    directives = "\n".join([directives] + block_sdfs)
    os.makedirs("directives", exist_ok=True)

    with open(
        "directives/bimanual_IIWA14_stackbot_and_assets.dmd.yaml", "w"
    ) as f:
        f.write(directives)

def generate_scenario_yaml():
    directives_main = f"file://{Path.cwd()}/directives/bimanual_IIWA14_stackbot_and_assets.dmd.yaml"
    directives_cams = f"file://{Path.cwd()}/directives/camera_directives.dmd.yaml"

    scenario_yaml = f"""directives:
- add_directives:
    file: {directives_main}
- add_directives:
    file: {directives_cams}

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

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
    """

    os.makedirs("scenarios", exist_ok=True)

    with open(
        "scenarios/bimanual_IIWA14_stackbot_assets_and_cameras.scenario.yaml",
        "w",
    ) as f:
        f.write(scenario_yaml)

def randomize_blocks_near_kuka(
    block_names,
    plant,
    context,
    base_model_name: str = "iiwa",
    base_link_name: str = "iiwa_link_0",
    min_radius: float = 0.35,
    max_radius: float = 0.75,
    fov_degrees: float = 100.0,  # cone around direction to platform
    platform_model_name: str = "platform",
    platform_link_name: str = "platform_link",
    platform_exclusion_radius: float = 0.30,
    max_tries_per_block: int = 100,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Randomizes blocks in an annulus [min_radius, max_radius] around the iiwa
    base, but only in a cone that points from the base toward the platform.
    Also avoids a disk around the platform (where we'll stack).

    Geometry is all in XY; Z and rotation are preserved.
    """
    if rng is None:
        rng = np.random.default_rng()

    base_instance = plant.GetModelInstanceByName(base_model_name)
    base_body = plant.GetBodyByName(base_link_name, base_instance)
    X_WBase = plant.EvalBodyPoseInWorld(context, base_body)
    p_WBase = X_WBase.translation()

    platform_instance = plant.GetModelInstanceByName(platform_model_name)
    platform_body = plant.GetBodyByName(platform_link_name, platform_instance)
    X_WPlat = plant.EvalBodyPoseInWorld(context, platform_body)
    p_WPlat = X_WPlat.translation()

    # platform footprint (known from how we created it)
    plat_size = np.array([0.5, 0.5, 0.05])
    plat_half = plat_size[:2] / 2.0

    # conservative block footprint (max of generated blocks)
    # w <= 0.10, l <= 0.30
    block_half = np.array([0.10, 0.30]) / 2.0

    # exclusion half-extent = platform half + block half + small margin
    margin = 0.02
    exclude_half = plat_half + block_half + margin

    # front = direction from base to platform in XY
    front_xy = p_WPlat[:2] - p_WBase[:2]
    if np.linalg.norm(front_xy) < 1e-6:
        front_xy = np.array([0.0, 1.0])  # fallback
    front_xy = front_xy / np.linalg.norm(front_xy)
    cos_fov_half = np.cos(np.deg2rad(fov_degrees) / 2.0)

    for name in block_names:
        instance = plant.GetModelInstanceByName(name)
        body_indices = list(plant.GetBodyIndices(instance))
        if not body_indices:
            raise RuntimeError(f"No bodies found for block model '{name}'")

        block_body = plant.get_body(body_indices[0])
        z = BLOCK_CENTER_Z

        for _ in range(max_tries_per_block):
            # sample radius
            r = np.sqrt(
                (max_radius**2 - min_radius**2) * rng.random()
                + min_radius**2
            )
            theta = 2.0 * np.pi * rng.random()
            dx, dy = r * np.cos(theta), r * np.sin(theta)
            candidate_xy = np.array([p_WBase[0] + dx, p_WBase[1] + dy])

            # must lie in cone pointing to platform
            v = candidate_xy - p_WBase[:2]
            if np.linalg.norm(v) < 1e-6:
                continue
            v = v / np.linalg.norm(v)
            cos_angle = np.dot(v, front_xy)
            if cos_angle < cos_fov_half:
                continue  # too far to the side / behind

            # must be outside platform exclusion radius
            if np.linalg.norm(candidate_xy - p_WPlat[:2]) < platform_exclusion_radius:
                continue

            delta = candidate_xy - p_WPlat[:2]
            if (np.abs(delta) <= exclude_half).all():
                # inside forbidden rectangle -> would intersect or be under platform
                continue

            # accept this sample :)
            p_WB_new = np.array([candidate_xy[0], candidate_xy[1], z])
            yaw = rng.uniform(0.0, 2.0 * np.pi)
            R_WB_new = RotationMatrix.MakeZRotation(yaw)
            X_WB_new = RigidTransform(R_WB_new, p_WB_new) # type: ignore
            plant.SetFreeBodyPose(context, block_body, X_WB_new)
            break
        else:
            print(
                f"[randomize_blocks_near_kuka] Warning: "
                f"could not find valid pose for {name} after {max_tries_per_block} tries."
            )