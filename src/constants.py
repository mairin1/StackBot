import numpy as np

BLOCK_COLOR_RGBA = np.array([0, 1, 0, 1])
FLOOR_TOP_Z = 0.0
BLOCK_HEIGHT = 0.06
BLOCK_CENTER_Z = FLOOR_TOP_Z + 0.5 * BLOCK_HEIGHT  # = 0.03

NUM_BLOCKS = 2

GRIPPER_OPEN = 0.2
GRIPPER_CLOSED = 0.

PLACE_Z_BUFFER = 0.02 # 0.05 def will work, to prevent collision of gripper with floor -- testing closer for greater stability
PICK_Z_CLEARANCE = 0.02 # >=0.03 is necessary for IK in some scenes, but more stable if we do closer

SIMULATOR_SETTLING_TIME = 5 # to let blocks setlle after dropping