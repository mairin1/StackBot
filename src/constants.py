import numpy as np

BLOCK_COLOR_RGBA = np.array([0, 1, 0, 1])
FLOOR_TOP_Z = 0.0
BLOCK_HEIGHT = 0.06
BLOCK_CENTER_Z = FLOOR_TOP_Z + 0.5 * BLOCK_HEIGHT  # = 0.03

NUM_BLOCKS = 2

GRIPPER_OPEN = 0.3
GRIPPER_CLOSED = 0.

PLACE_Z_BUFFER = 0.05 # to prevent collision of gripper with floor 