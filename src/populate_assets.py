import numpy as np

def create_block_sdf(
        model_name : str, 
        size : np.ndarray, 
        mass : float = 1, 
        pose : np.ndarray = [0, 0, 0, 0, 0, 0],
        rgba : np.ndarray = [1, 1, 1, 1]
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

block_color_rgba = np.array([0, 1, 0, 1])

# I want blocks to be 10 x (10 + 2i) x 8 cm
for i in range(11):
    w = 0.1 # i think 10cm is about our max gripper width
    l = 0.1 + 0.02 * i # 10 + i cm   
    h = 0.06 
    with open(f"assets/block{i}.sdf", "w+") as f:
        f.write(create_block_sdf(f"block{i}", [w, l, h], rgba=block_color_rgba))#rgba=np.array([197, 152, 214, 255]) / 255))

# I want a floor
with open("assets/floor.sdf", "w+") as f:
    f.write(create_block_sdf("floor", [3, 3, 0.1], rgba=[0.1, 0.1, 0.1, 1]))

# I want a platform to stack on
with open("assets/platform.sdf", "w+") as f:
    f.write(create_block_sdf("platform", [0.5, 0.5, 0.05]))
