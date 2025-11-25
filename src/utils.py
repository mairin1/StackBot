import numpy as np
from pydrake.all import (
    RotationMatrix,
    RigidTransform,
    MultibodyPlant
)

from pydrake.systems.framework import Context

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

def save_context(context : Context, filename : str):
    time = context.get_time()
    xc = context.get_continuous_state_vector().CopyToVector()
    xd = context.get_discrete_state(0).CopyToVector()

    np.savez(filename, time=time, xc=xc, xd=xd)

def load_context(context : Context, filename : str):
    with np.load(filename) as data:
        time = data["time"]
        xc = data["xc"]
        xd = data["xd"]
    
    context.SetTime(time)
    context.SetContinuousState(xc)
    context.SetDiscreteState(0, xd)