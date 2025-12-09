import numpy as np
from pydrake.all import (
    BasicVector,
    LeafSystem,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    TrajectorySource,
    RollPitchYaw
)

import numpy as np

def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero vector")
    return v / n
    
def _make_basis_from_approach_and_close(approach_W, close_W):
    """
    Returns orthonormal world basis (xW, yW, zW) where:
      - aW is approach direction (tool 'approach axis' in world)
      - cW is closing direction (tool 'closing axis' in world), projected perpendicular to aW
    """
    aW = _unit(approach_W)
    cW = close_W - np.dot(close_W, aW) * aW
    cW = _unit(cW)
    oW = np.cross(cW, aW)  # completes right-handed frame
    oW = _unit(oW)
    # re-orthogonalize cW (numerical)
    cW = _unit(np.cross(aW, oW))
    return oW, cW, aW  # (xW, yW, zW) for the "grasp frame"

def rotation_W_from_axes(approach_W, close_W, ee_approach_axis, ee_close_axis):
    """
    Build RotationMatrix R_WE such that:
      - the EE approach axis (one of x/y/z in EE) points along approach_W
      - the EE closing axis points along close_W (projected to be perpendicular to approach)
    """
    xG, yG, zG = _make_basis_from_approach_and_close(approach_W, close_W)  # world directions

    # desired world directions for EE axes = exW, eyW, ezW
    axis_to_vec = {}
    axis_to_vec[ee_approach_axis] = zG
    axis_to_vec[ee_close_axis] = yG

    remaining = {"x", "y", "z"} - {ee_approach_axis, ee_close_axis}
    rem_axis = next(iter(remaining))
    # x = y × z, y = z × x, z = x × y depending on which one is missing
    if rem_axis == "x":
        axis_to_vec["x"] = _unit(np.cross(axis_to_vec["y"], axis_to_vec["z"]))
    elif rem_axis == "y":
        axis_to_vec["y"] = _unit(np.cross(axis_to_vec["z"], axis_to_vec["x"]))
    else:  # rem_axis == "z"
        axis_to_vec["z"] = _unit(np.cross(axis_to_vec["x"], axis_to_vec["y"]))

    R = np.column_stack([axis_to_vec["x"], axis_to_vec["y"], axis_to_vec["z"]])
    return RotationMatrix(R)

class TrajectoryCommander(LeafSystem):
    """
    Outputs:
      - V_WG_des (6): spatial velocity [wx wy wz vx vy vz] in world
      - wsg_position (1): gripper opening command (meters)
    """
    def __init__(self, dt_fd: float = 1e-3):
        super().__init__()
        self._pose_traj = None
        self._V_traj = None
        self._wsg_traj = None
        self._dt_fd = float(dt_fd)
        self._t0 = 0.0 # time offset

        self.DeclareVectorOutputPort("V_WG_des", BasicVector(6), self._calc_V)
        self.DeclareVectorOutputPort("wsg_position", BasicVector(1), self._calc_wsg)

    def set_trajectories(self, pose_traj, wsg_traj):
        self._pose_traj = pose_traj
        self._wsg_traj = wsg_traj
        self._V_traj = None
        if pose_traj is not None:
            try:
                self._V_traj = pose_traj.MakeDerivative()
            except Exception:
                self._V_traj = None

    def set_time_offset(self, t0: float):
        self._t0 = float(t0)

    def end_time(self) -> float:
        if self._pose_traj is None:
            return 0.0
        return float(self._pose_traj.end_time())

    def _calc_wsg(self, context, output):
        if self._wsg_traj is None:
            output.SetFromVector([0.0])
            return
        t = context.get_time() - self._t0
        v = np.asarray(self._wsg_traj.value(t)).reshape(-1)
        output.SetFromVector([float(v[0])])

    def _calc_V(self, context, output):
        if self._pose_traj is None:
            output.SetFromVector(np.zeros(6))
            return
        t = float(context.get_time()) - self._t0
        v = np.asarray(self._V_traj.value(t)).reshape(-1) # type: ignore
        output.SetFromVector(v[:6])

# GRASP SELECTION

def design_top_down_grasp(
    X_WB: RigidTransform,
    extents_xyz: np.ndarray,
    ee_approach_axis,
    ee_close_axis,
    approach_clearance: float = 0.12,
):
    R_WB = X_WB.rotation().matrix()
    p_WB = X_WB.translation()
    ex, ey, ez = extents_xyz

    xW = R_WB[:, 0]
    yW = R_WB[:, 1]

    # pick the shorter horizontal axis for closing (more likely to fit)
    closeW = xW if ex < ey else yW
    # closeW = xW

    # project closing direction into world XY (keeps it horizontal)
    closeW = np.array([closeW[0], closeW[1], 0.0])
    if np.linalg.norm(closeW) < 1e-6:
        closeW = np.array([1.0, 0.0, 0.0])

    approachW = np.array([0.0, 0.0, -1.0])  # always straight down in world

    R_WE = rotation_W_from_axes(
        approach_W=approachW,
        close_W=closeW,
        ee_approach_axis=ee_approach_axis,
        ee_close_axis=ee_close_axis,
    )
    
    p_WE_pick = p_WB + np.array([0.0, 0.0, 0.5 * ez])
    X_WE_pick = RigidTransform(R_WE, p_WE_pick) # type: ignore
    X_WE_pre  = RigidTransform(R_WE, p_WE_pick + np.array([0.0, 0.0, approach_clearance])) # type: ignore

    return X_WE_pre, X_WE_pick

# TRAJECTORY BUILDERS

def make_pick_trajectories(X_WG_initial: RigidTransform,
                           X_WG_pre: RigidTransform,
                           X_WG_pick: RigidTransform,
                           lift_distance: float = 0.12):
    # lift pose: same orientation, translate up in world z
    p = X_WG_pick.translation().copy()
    X_WG_lift = RigidTransform(X_WG_pick.rotation(), p + np.array([0.0, 0.0, lift_distance]))

    # keyframes (pose)
    Xs = [X_WG_initial, X_WG_pre, X_WG_pick, X_WG_pick, X_WG_lift]
    ts = [0.0, 3.0, 6.0, 7.0, 10.0]

    pose_traj = PiecewisePose.MakeLinear(ts, Xs)

    try:
        V_traj = pose_traj.MakeDerivative()
        V_source = TrajectorySource(V_traj)
        use_derivative_source = True
    except Exception:
        use_derivative_source = False
        V_source = None

    # gripper trajectory: open → close → stay close
    opened = 0.15
    closed = 0.01
    finger = np.array([[opened, opened, opened, closed, closed]])
    wsg_traj = PiecewisePolynomial.FirstOrderHold(ts, finger)
    wsg_source = TrajectorySource(wsg_traj)

    return pose_traj, V_source, use_derivative_source, wsg_source, wsg_traj

def make_pick_and_place_trajectories(
        X_WG_initial: RigidTransform,
        X_WG_pre_pick: RigidTransform,
        X_WG_pick: RigidTransform,
        place_xy: np.ndarray,
        place_z: float,
        lift_distance: float = 0.5, 
        approach_clearance: float = 0.12
):
    """ 
    Build a single trajectory that:
        1) Starts from X_WG_initial
        2) Goes to pre-pick, then pick, closes, lifts
        3) Moves to pre-place over (place_xy, place_z), then down to place
        4) Opens gripper, retreats to pre-place
        5) Returns to X_WG_initial
    """
    place_xy = np.asarray(place_xy, dtype=float).reshape(2)


    rpy_pick = RollPitchYaw(X_WG_pick.rotation())
    R_place = RollPitchYaw(
        rpy_pick.roll_angle(),
        rpy_pick.pitch_angle(),
        45,
    ).ToRotationMatrix()

    p_lift = X_WG_pick.translation().copy()
    p_lift[2] += float(lift_distance)
    X_WG_lift = RigidTransform(X_WG_pick.rotation(), p_lift)

    p_place = np.array([place_xy[0], place_xy[1], place_z], dtype=float)
    X_WG_place = RigidTransform(R_place, p_place)
    X_WG_pre_place = RigidTransform(R_place, p_place + np.array([0.0, 0.0, approach_clearance], dtype=float))

    X_WG_final = X_WG_initial

    Xs = [
        X_WG_initial,   # t0: start
        X_WG_pre_pick,  # t1: pre-pick
        X_WG_pick,      # t2: descend to pick
        X_WG_pick,      # t3: close fingers
        X_WG_pick,      # t3.5: pause with closed fingers
        X_WG_lift,      # t4: lift up
        X_WG_pre_place, # t5: move above place
        X_WG_place,     # t6: descend to place
        X_WG_place,     # t7: open fingers
        X_WG_pre_place, # t8: retreat up
        X_WG_final,     # t9: go back home
    ]

    delta_ts = [
        0, # initial
        3, # pre-pick
        3, # pick
        2, # pause
        3, # close
        5, # lift
        3, # pre-place
        3, # place
        1, # open
        2, # retreat
        3  # home
    ]
    num_ts = len(delta_ts)
    ts = []
    count = 0
    for i in range(num_ts):
        count += delta_ts[i]
        ts.append(count)

    pose_traj = PiecewisePose.MakeLinear(ts, Xs)

    opened = 0.2
    closed = 0
    finger = np.array([[
        opened,  # t0
        opened,  # t1
        closed,  # t2
        closed,  # t3 (close)
        closed,  # t3.5 (close)
        closed,  # t4
        closed,  # t5
        closed,  # t6 (still closed at contact)
        opened,  # t7 (open to release)
        opened,  # t8
        opened,  # t9
    ]])

    wsg_traj = PiecewisePolynomial.FirstOrderHold(ts, finger)
    wsg_source = TrajectorySource(wsg_traj)

    return pose_traj, wsg_source, wsg_traj