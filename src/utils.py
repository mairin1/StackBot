import json
import numpy as np
from pydrake.all import (
    RotationMatrix,
    RigidTransform,
    MultibodyPlant,
    FixedOffsetFrame,
    PlanarJoint
)
from constants import *
from pydrake.systems.framework import Context

def randomize_blocks(blocks : list[str], plant, plant_context):
    '''
    deprecated
    '''

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
        
def create_randomized_block_directives(block_names : list[str]):
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
        z = 0.1
        rotation = np.random.randint(0, 360)

        directives +=f"""
    - add_model:
        name: {block_name}
        file: package://stackbot/{block_name}.sdf
        default_free_body_pose:
            {block_name}_link:
                translation: [{x}, {y}, {z}]
                rotation: !Rpy {{ deg: [0, 0, {rotation}]}}"""

    return directives

def create_block_directives_from_file(filename: str):
    with open(filename, "r") as f:
        positions = json.load(f)

    directives = ""
    
    for body_name, pose in positions.items():
        if "block" in body_name:
            block_name = body_name.split("_")[0] # hacky, block10_link -> block10
            directives +=f"""
    - add_model:
        name: {block_name}
        file: package://stackbot/{block_name}.sdf
        default_free_body_pose:
            {body_name}:
                translation: {str(pose["translation"])}
                rotation: !Rpy {{ deg: [0, 0, {pose["rotation"][2] * 180 / np.pi}]}}"""
    
    return directives


def save_positions(plant : MultibodyPlant, context : Context, body_names : list[str], filename : str):
    """
    more useful saving
    """
    positions = {}
    for body_name in body_names:
        body = plant.GetRigidBodyByName(body_name)
        pose = plant.GetFreeBodyPose(context, body)
        translation = pose.translation().tolist()
        rotation = pose.rotation().ToRollPitchYaw().vector().tolist()

        positions[body_name] = {
            "translation" : translation,
            "rotation" : rotation
        }
    
    with open(filename, "w") as f:
        json.dump(positions, f)


def save_context(context : Context, filename : str):
    """
    e.g.
    context = simulator.get_context()
    save_context(context, "../assets/contexts/filename.npz")
    """
    time = context.get_time()
    xc = context.get_continuous_state_vector().CopyToVector()
    xd = context.get_discrete_state(0).CopyToVector()

    np.savez(filename, time=time, xc=xc, xd=xd)

def load_context(context : Context, filename : str):
    """
    e.g.
    context = diagram.CreateDefaultContext()
    load_context(context, "../assets/contexts/filename.npz")
    simulator = Simulator(diagram, context)
    """
    with np.load(filename) as data:
        time = data["time"]
        xc = data["xc"]
        xd = data["xd"]
    
    context.SetTime(time)
    context.SetContinuousState(xc)
    context.SetDiscreteState(0, xd)