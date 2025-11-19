import numpy as np
from pydrake.all import (
    RotationMatrix,
    RigidTransform,
    MultibodyPlant
)

def randomize_blocks(blocks : list[str], plant, plant_context):

    def sample_drop_point(inner_d = 0.2, outer_d = 0.8):
        """
        sample a point on the xy plane such that |x|, |y| > inner_d, |x|, |y| < outer_d
        (think concentric squares)
        """
        x, y = np.random.uniform(low=inner_d, high=outer_d, size=2) * np.random.choice([-1, 1], size=2)
        z = np.random.uniform(0.5, 0.8)
        return x, y, z

    for block in blocks:
        body = plant.GetBodyByName(f"{block}_link")
        position = sample_drop_point()
        rotation = RotationMatrix.MakeZRotation(np.random.uniform(0, 2*np.pi))
        plant.SetFreeBodyPose(
            plant_context,
            body,
            RigidTransform(rotation, position)
        )