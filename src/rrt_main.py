# run from StackBot/ as python src/rrt_main.py --seed 114 (replace with whatever seed you want)
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    PiecewisePolynomial,
    Simulator,
    StartMeshcat,
    ConstantVectorSource,
    RollPitchYaw,
    TrajectorySource,
    RigidTransform
)

from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.meshcat_utils import AddMeshcatTriad

from constants import *
from scene_utils import generate_scenario_yaml, get_block_poses
from perception_utils import get_point_cloud_from_cameras, perceive
from planning_utils import design_top_down_grasp
from utils import *
from eval_utils import *

from rrt_planner import pick_and_place_traj_rrt_one_block

import argparse
import json

def execute_rrt_path(full_q_path, full_g_path, scenario, block_poses: dict[str, "RigidTransform"], meshcat, segment_ranges, time_offset: float):
    """
    Build a fresh station for execution and feed the RRT joint/gripper
    trajectories directly into iiwa.position and wsg.position.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeHardwareStation(
            scenario=scenario,
            meshcat=meshcat,
            package_xmls=["assets/models/package.xml"],
        )
    )

    plant = station.GetSubsystemByName("plant")
    assert len(full_q_path) == len(full_g_path)

    # placement related segments
    super_slow_segments = {2, 6}
    slow_segments = {3, 5, 7}
    
    # segments that can go fast
    fast_segments = {}

    base_dt = 0.05
    super_slow_factor = 50
    slow_factor = 10.0
    fast_factor = 0.8
    ts_local = [0.0]
    print(segment_ranges)
    for seg_idx, (start, end) in enumerate(segment_ranges):
        num_steps = end - start
        if seg_idx in slow_segments:
            dt_seg = base_dt * slow_factor
        elif seg_idx in fast_segments:
            dt_seg = base_dt * fast_factor
        elif seg_idx in super_slow_segments:
            dt_seg = base_dt * super_slow_factor
            print("super slow time = ", dt_seg)
        else:
            dt_seg = base_dt
        for _ in range(num_steps):
            ts_local.append(ts_local[-1] + dt_seg)

    ts_local = np.array(ts_local[:len(full_q_path)])
    duration = float(ts_local[-1])
    ts = ts_local + time_offset # to accumulate across blocks
    
    # joint path: shape (7, N)
    Q = np.vstack(full_q_path).T
    q_traj = PiecewisePolynomial.FirstOrderHold(ts, Q)

    # gripper path: shape (1, N)
    G = np.array(full_g_path, dtype=float).reshape(1, -1)
    g_traj = PiecewisePolynomial.FirstOrderHold(ts, G)

    q_source = builder.AddSystem(TrajectorySource(q_traj))
    g_source = builder.AddSystem(TrajectorySource(g_traj))

    builder.Connect(q_source.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(g_source.get_output_port(), station.GetInputPort("wsg.position"))

    # zero other inputs if they exist
    def connect_if_present(port_name, value):
        try:
            port = station.GetInputPort(port_name)
        except Exception:
            return
        src = builder.AddSystem(ConstantVectorSource(value))
        builder.Connect(src.get_output_port(), port)

    n_q = 7
    connect_if_present("iiwa.velocity", np.zeros(n_q))
    connect_if_present("iiwa.feedforward_torque", np.zeros(n_q))
    connect_if_present("wsg.force_limit", np.array([40.0]))

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)

    # apply shared block poses
    for block_name, X_WB in block_poses.items():
        model_inst = plant.GetModelInstanceByName(block_name)
        body = plant.GetBodyByName(f"{block_name}_link", model_inst)
        plant.SetFreeBodyPose(plant_context, body, X_WB)

    context.SetTime(time_offset)
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(ts[-1]) # ts[-1] = time_offset+duration here TODO verify!! -> yep
    print("Finished executing RRT path.")
    updated_block_poses: dict[str, RigidTransform] = {}
    for block_name in block_poses.keys():
        model_inst = plant.GetModelInstanceByName(block_name)
        body = plant.GetBodyByName(f"{block_name}_link", model_inst)
        X_WB_final = plant.EvalBodyPoseInWorld(plant_context, body)
        updated_block_poses[block_name] = X_WB_final

    return updated_block_poses, duration

def pick_block(
    estimated_X_WB,
    plant,
    plant_context,
    meshcat,
    place_xy: np.ndarray,
    place_z: float,
    block_poses: dict[str, "RigidTransform"],
    scenario
):
    """
    Use perception (estimated_X_WBs) to design the grasp and then call
    pick_traj_rrt_one_block to get a joint-space RRT path for one block.
    """
    # block_name = f"block{block_idx}"
    # block_inst = plant.GetModelInstanceByName(block_name)
    # block_body = plant.GetBodyByName(f"{block_name}_link", block_inst)
    # X_WB_true = plant.EvalBodyPoseInWorld(plant_context, block_body)

    X_WB_hat = estimated_X_WB[0]

    # extents estimate (you may want to tighten this later)
    extents_hat = (0.08, estimated_X_WB[1], 0.06)
    print("Estimated extents:", extents_hat)

    X_WG_pre, X_WG_pick = design_top_down_grasp(
        X_WB_hat,
        extents_hat,
        ee_approach_axis="y",
        ee_close_axis="x",
        z_clearance=0.03
    )

    AddMeshcatTriad(meshcat, "X_WB_hat", X_PT=X_WB_hat, length=0.15)
    AddMeshcatTriad(meshcat, "X_WG_pre", X_PT=X_WG_pre, length=0.15)
    AddMeshcatTriad(meshcat, "X_WG_pick", X_PT=X_WG_pick, length=0.15)

    # initial gripper pose in the perception/station world
    ee_body = plant.GetBodyByName("body")
    X_WG_initial = plant.EvalBodyPoseInWorld(plant_context, ee_body)

    # pick_traj_rrt_one_block constructs its own RRT world and return the path
    full_q_path, full_g_path, segment_ranges = pick_and_place_traj_rrt_one_block(
        scenario,
        X_WG_initial,
        X_WG_pre,
        X_WG_pick,
        place_xy,
        place_z,
        lift_distance=0.3,
        approach_clearance=0.12,
        align_stack_yaw=True,
        stack_yaw=0,
        max_rrt_iters=3000,
        meshcat=meshcat,
        verbose=True,
        block_poses=block_poses,
    )

    return full_q_path, full_g_path, segment_ranges


# Z_BUFFER = 0.05 # to prevent collision of gripper with floor 
if __name__ == "__main__":
    # successful_seeds = []
    # for seed in range(100, 120):
        # try: 
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="input seed")
    args = parser.parse_args()
    seed_input = args.seed if args.seed is not None else 114

    # randomize blocks once in this world
    rng = np.random.default_rng(seed=seed_input)
    num_blocks = rng.choice([4, 5, 6])
    block_numbers = rng.choice(np.arange(11), size=num_blocks, replace=False)
    print("This scenario uses blocks:", block_numbers)
    blocks = [f"block{i}" for i in block_numbers]
    NUM_BLOCKS = len(blocks)

    scenario = LoadScenario(
        data=generate_scenario_yaml(blocks, rng)
    )

    # build a diagram for perception and block randomization
    meshcat = StartMeshcat()
    print("Click the link above to open Meshcat (perception + planning).")

    builder = DiagramBuilder()
    # scenario = LoadScenario(filename=scenario_stack_file)
    station = builder.AddSystem(
        MakeHardwareStation(
            scenario=scenario,
            meshcat=meshcat,
            package_xmls=["assets/models/package.xml"],
        )
    )

    plant = station.GetSubsystemByName("plant")

    # add cameras / point clouds
    pc_systems = AddPointClouds(scenario=scenario, station=station, builder=builder)

    builder.ExportOutput(pc_systems["camera0"].get_output_port(), "camera0_point_cloud")
    builder.ExportOutput(pc_systems["camera1"].get_output_port(), "camera1_point_cloud")
    builder.ExportOutput(pc_systems["camera2"].get_output_port(), "camera2_point_cloud")

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    block_poses = get_block_poses(plant, plant_context, blocks)
    # publish once so camera clouds exist
    diagram.ForcedPublish(diagram_context)
    input("Hit enter to continue") # add this if you want to stop to look at scene before motion planning starts

    meshcat.StartRecording()

    # compute platform pose for placing
    platform_inst = plant.GetModelInstanceByName("platform")
    platform_body = plant.GetBodyByName("platform_link", platform_inst)
    X_WPlat = plant.EvalBodyPoseInWorld(plant_context, platform_body)
    p_WPlat = X_WPlat.translation()
    place_xy = p_WPlat[:2]
    platform_half_h = 0.05 / 2.0  # from create_block_sdf for platform
    block_h = BLOCK_HEIGHT

    # Perception
    point_cloud = get_point_cloud_from_cameras(diagram, diagram_context)
    est_X_WBs_by_length = perceive(point_cloud, meshcat)

    current_time = 0.0
    for stack_level in range(NUM_BLOCKS):
        print("---- Block", stack_level + 1, "----")
        print("platform_half_h =", platform_half_h)
        print("block_h =", block_h)

        place_z = platform_half_h + (stack_level + 1 + 0.5) * block_h + PLACE_Z_BUFFER
        print("place_z:", place_z)

        try: # get rid of this later, this is just for testing purposes
            full_q_path, full_g_path, segment_ranges = pick_block(
                estimated_X_WB=est_X_WBs_by_length[stack_level],
                plant=plant,
                plant_context=plant_context,
                meshcat=meshcat,
                place_xy=place_xy,
                place_z=place_z,
                block_poses=block_poses,
                scenario=scenario
            )
        except IndexError:
            continue

        # execute that RRT path in a clean execution diagram
        block_poses, duration = execute_rrt_path(full_q_path, full_g_path, scenario, block_poses, meshcat, segment_ranges, time_offset=current_time)
        current_time += duration
        
        # input("Finished execution for this block. Press Enter for next (or Ctrl+C to stop)...\n") 
    # once all blocks done, stop and publish the full recording
    meshcat.StopRecording()
    meshcat.PublishRecording()
    yesNo = input("Enter Y to save the final stack poses: ")
    if (yesNo == "Y" or yesNo == "y"):
        filename = f"assets/contexts/stack_rrt_{seed_input}.json"
        # save_positions(plant, plant_context, [f"{name}_link" for name in blocks], filename)
        positions = {}
        for name in block_poses:
            pose = block_poses[name]
            translation = pose.translation().tolist()
            rotation = pose.rotation().ToRollPitchYaw().vector().tolist()

            positions[name] = {
                "translation" : translation,
                "rotation" : rotation
            }
        with open(filename, "w") as f:
            json.dump(positions, f)
        print(f"Translation Error: {avg_translation_error(filename)}" )
        print(f"Rotation Error: {avg_z_rotation_error(filename, 0)}" )
    input("All blocks done. Press Enter to close Meshcat...\n")
    #     except:
    #         continue
    #     # seed succeeded
    #     print(f"seed succeeded: {seed}")
    #     successful_seeds.append(seed)
    
    # print(successful_seeds)
