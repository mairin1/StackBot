import json
import numpy as np

def avg_translation_error(filename):
    with open(filename, "r") as f:
        block_poses = json.load(f)
    total_offset = 0
    for block in block_poses.values():
        translation = block["translation"]
        total_offset += (translation[0] ** 2 + translation[1] ** 2) ** 0.5
    return total_offset / len(block_poses)

def avg_z_rotation_error(filename, target_rot):
    with open(filename, "r") as f:
        block_poses = json.load(f)
    total_error = 0
    for block in block_poses.values():
        theta = block["rotation"][2]
        total_error += abs((theta + np.pi/2) % np.pi - np.pi/2 - target_rot)
    return total_error / len(block_poses)