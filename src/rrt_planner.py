# stackbot_rrt_main.py
from typing import List, Tuple
import numpy as np

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    MultibodyPlant,
    InverseKinematics,
    Solve,
    RollPitchYaw
)
from pydrake.all import StartMeshcat

from constants import *
from rrt_utils import *


def solve_ik_for_pose(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    theta_bound: float = 0.05 * np.pi, # was 0.01
    pos_tol: float = 0.015, # was 0.015
    q_nominal: np.ndarray | None = None,
) -> Tuple[float, ...]:
    """
    Solve IK to find a 7-DOF iiwa configuration that reaches X_WG_target.
    """

    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")

    ik = InverseKinematics(plant)
    q_vars = ik.q()[:7]
    prog = ik.prog()

    # Orientation constraint
    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=X_WG_target.rotation(),
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),
        theta_bound=theta_bound,
    )

    # Position constraint
    p_WG_target = X_WG_target.translation()
    ik.AddPositionConstraint(
        frameA=world_frame,
        frameB=gripper_frame,
        p_BQ=np.zeros(3),
        p_AQ_lower=p_WG_target - pos_tol * np.ones(3),
        p_AQ_upper=p_WG_target + pos_tol * np.ones(3),
    )

    if q_nominal is None:
        q_nominal = np.zeros(7)
    prog.AddQuadraticErrorCost(np.eye(7), q_nominal, q_vars)
    prog.SetInitialGuess(q_vars, q_nominal)

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("IK did not succeed for target pose")

    q_sol = result.GetSolution(q_vars)
    return tuple(float(x) for x in q_sol)


def plan_rrt_chain(
    sim: ManipulationStationSim,
    q_waypoints: List[Tuple[float, ...]],
    g_waypoints: List[float],
    max_iter_segment: int = 3000,
):
    """ RRT helper to chain several joint waypoints
    Given a list of joint waypoints [q0, q1, ..., qK], plan an RRT-Connect
    path between each consecutive pair and concatenate.
    """
    assert len(q_waypoints) == len(g_waypoints)

    full_q_path: List[Tuple[float, ...]] = []
    full_g_path: List[float] = []
    segment_ranges: List[Tuple[int, int]] = []

    for i in range(len(q_waypoints) - 1):
        q_start = q_waypoints[i]
        q_goal = q_waypoints[i + 1]
        g_seg = g_waypoints[i]  # constant gripper for this motion segment

        if np.allclose(q_start, q_goal, atol=1e-8):
            path_seg = [q_start, q_goal]  # constant arm pose
            iters = 0
        else: 
            tools = RRT_Connect_tools(sim, start=q_start, goal=q_goal)

            # debugging - check straight-line path
            direct = tools.calc_intermediate_qs_wo_collision(q_start, q_goal, g_seg)
            # print(f"Segment {i}: direct path length = {len(direct)}")

            if len(direct) > 1:
                # print(f"Segment {i}: using direct straight-line path (no RRT needed)")
                path_seg, iters = direct, 0
            else:
                path_seg, iters = rrt_connect_planning(
                    sim, q_start, q_goal, max_iterations=max_iter_segment, gripper_setpoint=g_seg
                )
            # print(f"Segment {i}: RRT iterations = {iters}, path length = {0 if path_seg is None else len(path_seg)}")

        if path_seg is None:
            raise RuntimeError(f"RRT-Connect failed between waypoint {i} and {i+1}")

        start_idx = len(full_q_path)
        if i == 0:
            full_q_path.extend(path_seg)
            full_g_path.extend([g_seg] * len(path_seg))
        else:
            # avoid duplicate seam
            if full_q_path[-1] == path_seg[0]:
                full_q_path.extend(path_seg[1:])
                full_g_path.extend([g_seg] * (len(path_seg) - 1))
            else:
                full_q_path.extend(path_seg)
                full_g_path.extend([g_seg] * len(path_seg))
        end_idx = len(full_q_path)
        segment_ranges.append((start_idx, end_idx))

    return full_q_path, full_g_path, segment_ranges

def pick_and_place_traj_rrt_one_block(
        scenario,
        X_WG_initial: RigidTransform,
        X_WG_pre: RigidTransform,
        X_WG_pick: RigidTransform,
        place_xy: np.ndarray,
        place_z: float,
        lift_distance: float = 0.3,
        approach_clearance: float = 0.12,
        align_stack_yaw: bool = True,
        stack_yaw: float = 0,
        max_rrt_iters: int = 3000,
        meshcat = None,
        verbose = True,
        block_poses: dict[str, "RigidTransform"] | None = None,
    ):
    if meshcat is None:
        meshcat = StartMeshcat()
        print("Click the link above to open Meshcat in your browser (RRT demo).")

    sim = ManipulationStationSim(
        scenario=scenario,
        q_iiwa=None, # TODO: should in theory match with X_WG_initial
        gripper_setpoint=GRIPPER_OPEN,
        meshcat=meshcat,
        block_poses=block_poses,
    )
    plant = sim.plant
    place_xy = np.asarray(place_xy, dtype=float).reshape(2)

    p_lift = X_WG_pick.translation().copy()
    p_lift[2] += lift_distance 
    X_WG_lift = RigidTransform(X_WG_pick.rotation(), p_lift)

    R_pick = X_WG_pick.rotation()
    if align_stack_yaw:
        rpy_pick = RollPitchYaw(R_pick)
        R_place = RollPitchYaw(
            rpy_pick.roll_angle(),
            rpy_pick.pitch_angle(),
            stack_yaw,
        ).ToRotationMatrix()
    else:
        R_place = R_pick

    p_place = np.array([place_xy[0], place_xy[1], place_z], dtype=float)
    X_WG_place = RigidTransform(R_place, p_place)
    X_WG_pre_place = RigidTransform(R_place, p_place + np.array([0.0, 0.0, approach_clearance], dtype=float))
    X_WG_final = X_WG_initial
    
    q_initial = solve_ik_for_pose(plant, X_WG_initial)
    q_pre = solve_ik_for_pose(plant, X_WG_pre, q_nominal=np.array(q_initial))
    q_pick = solve_ik_for_pose(plant, X_WG_pick, q_nominal=np.array(q_pre))
    q_close = q_pick
    q_lift = solve_ik_for_pose(plant, X_WG_lift, q_nominal=np.array(q_close))
    q_pre_place = solve_ik_for_pose(plant, X_WG_pre_place, q_nominal=np.array(q_lift))
    q_place = solve_ik_for_pose(plant, X_WG_place, q_nominal=np.array(q_pre_place))
    q_open = q_place
    q_post_place = solve_ik_for_pose(plant, X_WG_pre_place, q_nominal=np.array(q_open))
    q_final = solve_ik_for_pose(plant, X_WG_final, q_nominal=np.array(q_open))

    if verbose:
        print("Collision status:")
        print("  q_initial:", sim.ExistsCollision(q_initial, GRIPPER_OPEN))
        print("  q_pre:", sim.ExistsCollision(q_pre, GRIPPER_OPEN))
        print("  q_pick:", sim.ExistsCollision(q_pick, GRIPPER_CLOSED))
        print("  q_close:", sim.ExistsCollision(q_close, GRIPPER_CLOSED))
        print("  q_lift:", sim.ExistsCollision(q_lift, GRIPPER_CLOSED))
        print("  q_pre_place:", sim.ExistsCollision(q_pre_place, GRIPPER_CLOSED))
        print("  q_place:", sim.ExistsCollision(q_place, GRIPPER_CLOSED))
        print("  q_open:", sim.ExistsCollision(q_open, GRIPPER_OPEN))
        print("  q_post_place:", sim.ExistsCollision(q_post_place, GRIPPER_OPEN))
        print("  q_final:", sim.ExistsCollision(q_final, GRIPPER_OPEN))
        # 0: qi-qpre, 1: qpre-qpick, 2: qpick-qc, 3: qc-ql, 4: ql-qpp, 5: qpp-qp, 6: qp-qo, 7: qo-qpp, 8: qpp-qf

    q_waypoints = [q_initial, q_pre, q_pick, q_close, q_lift, q_pre_place, q_place, q_open, q_post_place, q_final]
    g_waypoints = [   # mapping to arm q0 -> q1 motion
        GRIPPER_OPEN, # qi-qpre
        GRIPPER_OPEN, # qpre-qpick
        GRIPPER_CLOSED, # qpick-qc
        GRIPPER_CLOSED, # qc-ql
        GRIPPER_CLOSED, # ql-qpp
        GRIPPER_CLOSED, # qpp-p
        GRIPPER_OPEN, # qp-qo
        GRIPPER_OPEN, # qo-qpp
        GRIPPER_OPEN, # qpp-qf
        GRIPPER_OPEN # this is actually never used -- unless we want to start doing linear interpolation between gripper endpoints
    ]

    print("Planning RRT-Connect path...")
    full_q_path, full_g_path, segment_ranges = plan_rrt_chain(sim, q_waypoints, g_waypoints, max_iter_segment=max_rrt_iters)
    print(f"Total path length: {len(full_q_path)} configurations")

    return full_q_path, full_g_path, segment_ranges
