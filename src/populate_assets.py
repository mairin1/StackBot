import numpy as np

def create_block_sdf(
        model_name : str, 
        size : np.ndarray, 
        mass : float = 1, 
        pose : np.ndarray = [0, 0, 0, 0, 0, 0],
        rgba : np.ndarray = [1, 1, 1, 1],
        mu_static: float = 1.0,
        mu_dynamic: float = 1.0,
) -> str:

    assert len(pose) == 6, "pose must be 6d"
    assert len(size) == 3, "size must be 3d"

    a, b, c = size
    ixx = mass * (b**b + c**c) / 12
    iyy = mass * (a**a + c**c) / 12
    izz = mass * (a**a + b**b) / 12

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

def create_cylinder_sdf(name, radius, length, rgba = [1, 1, 1, 1]):
    return f"""
    <?xml version="1.0" ?>
<sdf version="1.5">
  <model name="{name}">
    <link name="{name}_link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <diffuse>{" ".join(str(num) for num in rgba)}</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <surface>
            <friction>
                <ode>
                    <mu>2.0</mu>
                    <mu2>1.5</mu2>
                </ode>
            </friction>
        </surface>
      </collision>
    </link>
  </model>
</sdf>"""

block_color_rgba = np.array([0, 1, 0, 1])

folder = "assets/models"

# I want blocks to be 10 x (10 + 2i) x 8 cm
for i in range(11):
    w = 0.06# i think 10cm is about our max gripper width
    l = 0.1 + 0.02 * i # 10 + i cm   
    h = 0.06 
    with open(f"{folder}/block{i}.sdf", "w+") as f:
        f.write(create_block_sdf(f"block{i}", [w, l, h], mass=0.2, rgba=block_color_rgba))#rgba=np.array([197, 152, 214, 255]) / 255))

# I want a floor
with open(f"{folder}/floor.sdf", "w+") as f:
    f.write(create_block_sdf("floor", [4, 4, 0.1], rgba=[0.1, 0.1, 0.1, 1]))

# I want a platform to stack on
with open(f"{folder}/platform.sdf", "w+") as f:
    f.write(create_block_sdf("platform", [0.5, 0.5, 0.05]))

# I want an iiwa probe
with open(f"{folder}/probe.sdf", "w+") as f:
    f.write(create_block_sdf("probe", [0.02, 0.02, 0.2]))


# round variants
with open(f"{folder}/round_platform.sdf", "w+") as f:
    f.write(create_cylinder_sdf("platform", 0.3, 0.05))

with open(f"{folder}/round_floor.sdf", "w+") as f:
    f.write(create_cylinder_sdf("floor", 2, 0.1, rgba=[0.1, 0.1, 0.1, 1]))

# obstacle

with open(f"{folder}/post.sdf", "w+") as f:
    f.write(create_block_sdf("post", [0.05, 0.05, 0.5]))