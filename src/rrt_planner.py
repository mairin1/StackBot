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


def _solve_ik_single(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    theta_bound: float,
    pos_tol: float,
    q_nominal_iiwa: np.ndarray | None = None,
    plant_context=None,  # kept for symmetry, but not used here
) -> Tuple[float, ...]:
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")

    ik = InverseKinematics(plant)
    prog = ik.prog()
    q_vars = ik.q()[:7]

    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=X_WG_target.rotation(),
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),
        theta_bound=theta_bound,
    )

    p_WG_target = X_WG_target.translation()
    ik.AddPositionConstraint(
        frameA=world_frame,
        frameB=gripper_frame,
        p_BQ=np.zeros(3),
        p_AQ_lower=p_WG_target - pos_tol * np.ones(3),
        p_AQ_upper=p_WG_target + pos_tol * np.ones(3),
    )

    if q_nominal_iiwa is None:
        q_nominal_iiwa = np.zeros(7)

    prog.AddQuadraticErrorCost(np.eye(7), q_nominal_iiwa, q_vars)
    prog.SetInitialGuess(q_vars, q_nominal_iiwa)

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError(f"IK failed (theta_bound={theta_bound})")

    q_sol = result.GetSolution(q_vars)
    return tuple(float(x) for x in q_sol)


def solve_ik_for_pose(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    theta_bound_feasible: float = 0.08 * np.pi,  # coarse
    theta_bound_refine: float = 0.01 * np.pi,  # tight
    pos_tol: float = 0.015,
    q_nominal_iiwa: np.ndarray | None = None,
    plant_context = None,
) -> Tuple[float, ...]:
    """
    Two-stage IK:
      1) Find a feasible solution with a looser orientation bound.
      2) Try to refine with a tighter orientation bound starting from that solution.
    """

    # stage 1: coarse / feasible solve
    q_coarse = _solve_ik_single(
        plant=plant,
        X_WG_target=X_WG_target,
        theta_bound=theta_bound_feasible,
        pos_tol=pos_tol,
        q_nominal_iiwa=q_nominal_iiwa,
        plant_context=plant_context,
    )

    # stage 2: refinement solve with tighter theta_bound, using coarse solution as nominal
    try:
        q_refined = _solve_ik_single(
            plant=plant,
            X_WG_target=X_WG_target,
            theta_bound=theta_bound_refine,
            pos_tol=pos_tol,
            q_nominal_iiwa=np.array(q_coarse),
            plant_context=plant_context,
        )
        return q_refined
    except RuntimeError:
        # Could not satisfy the 0.01*pi bound; fall back to the coarse solution
        print("Refine IK (theta_bound=0.01π) failed; using 0.03π solution instead.")
        return q_coarse


# def solve_ik_for_pose(
#     plant: MultibodyPlant,
#     X_WG_target: RigidTransform,
#     theta_bound: float = 0.03 * np.pi, # was 0.01
#     pos_tol: float = 0.015, # was 0.015
#     q_nominal_iiwa: np.ndarray | None = None,
#     plant_context = None # here as placeholder to make below code easier
# ) -> Tuple[float, ...]:
#     """
#     Solve IK to find a 7-DOF iiwa configuration that reaches X_WG_target.
#     """

#     world_frame = plant.world_frame()
#     gripper_frame = plant.GetFrameByName("body")

#     ik = InverseKinematics(plant)
#     q_vars = ik.q()[:7]
#     prog = ik.prog()

#     # Orientation constraint
#     ik.AddOrientationConstraint(
#         frameAbar=world_frame,
#         R_AbarA=X_WG_target.rotation(),
#         frameBbar=gripper_frame,
#         R_BbarB=RotationMatrix(),
#         theta_bound=theta_bound,
#     )

#     # Position constraint
#     p_WG_target = X_WG_target.translation()
#     ik.AddPositionConstraint(
#         frameA=world_frame,
#         frameB=gripper_frame,
#         p_BQ=np.zeros(3),
#         p_AQ_lower=p_WG_target - pos_tol * np.ones(3),
#         p_AQ_upper=p_WG_target + pos_tol * np.ones(3),
#     )

#     if q_nominal_iiwa is None:
#         q_nominal_iiwa = np.zeros(7)
#     prog.AddQuadraticErrorCost(np.eye(7), q_nominal_iiwa, q_vars)
#     prog.SetInitialGuess(q_vars, q_nominal_iiwa)

#     result = Solve(prog)
#     if not result.is_success():
#         raise RuntimeError("IK did not succeed for target pose")

#     q_sol = result.GetSolution(q_vars)
#     return tuple(float(x) for x in q_sol)

def solve_ik_for_pose_collisionavoid(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    plant_context,
    theta_bound: float = 0.03 * np.pi,
    pos_tol: float = 0.015,
    q_nominal_iiwa: np.ndarray | None = None,
    min_distance: float = 0.01,
    max_tries: int = 100,
) -> tuple[float, ...]:
    """
    Collision-aware IK for the iiwa gripper pose X_WG_target.

    - Uses world frame and gripper frame "body"
    - Enforces:
        * position within pos_tol box
        * orientation within theta_bound
        * minimum signed distance between all geometries >= min_distance
    - Returns a 7-tuple of iiwa joint angles.
    """

    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")

    # build IK with a context, so distance queries see the right geometry config
    ik = InverseKinematics(plant, plant_context)
    prog = ik.prog()
    q_vars = ik.q()
    n_q = len(q_vars)

    # start from current plant positions as default
    q_nominal_full = plant.GetPositions(plant_context).copy()
    if q_nominal_iiwa is not None:
        # overwrite first 7 (assumed iiwa) with provided nominal
        q_nominal_full[:7] = q_nominal_iiwa

    # quadratic cost to stay near nominal
    prog.AddQuadraticErrorCost(np.eye(n_q), q_nominal_full, q_vars)

    # POSE CONSTRAINTS ON GRIPPER

    # position: gripper origin within pos_tol box around target translation
    p_WG = X_WG_target.translation()
    ik.AddPositionConstraint(
        frameA=world_frame,
        frameB=gripper_frame,
        p_BQ=np.zeros(3),
        p_AQ_lower=p_WG - pos_tol * np.ones(3),
        p_AQ_upper=p_WG + pos_tol * np.ones(3),
    )

    # orientation: bound angle between desired and actual to theta_bound
    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=X_WG_target.rotation(),  # desired rotation of world
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),  # identity in gripper frame
        theta_bound=theta_bound,
    )

    # COLLISION (MIN DIST) CONSTRAINT
    # require that the minimum signed distance between *all* pairs of geometries is at least min_distance (0.01 = 1 cm)
    ik.AddMinimumDistanceLowerBoundConstraint(min_distance)

    # joint limits for random init guesses
    lo = plant.GetPositionLowerLimits().copy()
    hi = plant.GetPositionUpperLimits().copy()
    for i in range(len(lo)):
        if (not np.isfinite(lo[i])) or (not np.isfinite(hi[i])) or (hi[i] - lo[i] > 1e6):
            lo[i], hi[i] = -np.pi, np.pi

    # try multiple random guesses
    for _ in range(max_tries):
        q0 = np.random.uniform(lo, hi)
        # keep nominal iiwa close to what we want (helps convergence)
        q0[:7] = q_nominal_full[:7]
        prog.SetInitialGuess(q_vars, q0)

        result = Solve(prog)
        if result.is_success():
            q_sol_full = result.GetSolution(q_vars)
            # update context in case caller wants it consistent
            plant.SetPositions(plant_context, q_sol_full)
            # return only iiwa joints
            return tuple(float(x) for x in q_sol_full[:7])

    raise RuntimeError("Collision-aware IK did not succeed for target pose")


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
        avoid_obstacles: bool = False
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
    diagram = sim.diagram  # if your class doesn’t expose this, add a .diagram attribute there
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

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
    
    if avoid_obstacles:
        ik_function = solve_ik_for_pose_collisionavoid
    else:
        ik_function = solve_ik_for_pose
    q_initial = ik_function(plant, X_WG_initial, plant_context=plant_context)
    q_pre = ik_function(plant, X_WG_pre, q_nominal_iiwa=np.array(q_initial), plant_context=plant_context)
    q_pick = ik_function(plant, X_WG_pick, q_nominal_iiwa=np.array(q_pre), plant_context=plant_context)
    q_close = q_pick
    q_lift = ik_function(plant, X_WG_lift, q_nominal_iiwa=np.array(q_close), plant_context=plant_context)
    q_pre_place = ik_function(plant, X_WG_pre_place, q_nominal_iiwa=np.array(q_lift), plant_context=plant_context)
    q_place = ik_function(plant, X_WG_place, q_nominal_iiwa=np.array(q_pre_place), plant_context=plant_context)
    q_open = q_place
    q_post_place = ik_function(plant, X_WG_pre_place, q_nominal_iiwa=np.array(q_open), plant_context=plant_context)
    q_final = ik_function(plant, X_WG_final, q_nominal_iiwa=np.array(q_open), plant_context=plant_context)

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
