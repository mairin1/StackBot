import numpy as np

def create_block_sdf(model_name : str, size : np.ndarray, mass : float = 1, pose : np.ndarray = [0, 0, 0, 0, 0, 0]) -> str:
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
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")


for i in range(11):
    size = 0.01 * i # i cm     
    with open(f"assets/block{i}.sdf", "w+") as f:
        f.write(create_block_sdf(f"block{i}", [size, size, size]))