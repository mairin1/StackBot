# run this file from root -- python src/main.py (this is necessary for the paths to assets etc. to be correct)
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from pydrake.all import(
    DiagramBuilder,
    Integrator,
    PointCloud,
    Rgba,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    ConstantVectorSource
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation
)

from manipulation import running_as_notebook

from constants import *
from scene_utils import *
from perception_utils import *
from planning_utils import *
from diff_ik import PseudoInverseDiffIK

def connect_if_present(builder, station, port_name, value):
    try:
        port = station.GetInputPort(port_name)
    except Exception:
        return
    src = builder.AddSystem(ConstantVectorSource(value))
    builder.Connect(src.get_output_port(), port)

def generate_setup():
    meshcat = StartMeshcat()
    print("Click the link above to open Meshcat in your browser!")

    rng = np.random.default_rng(seed=1)

    block_numbers = rng.choice(range(11), size=rng.choice([4, 5, 6]), replace=False)
    print("This scenario uses blocks:", block_numbers)
    blocks = [f"block{i}" for i in block_numbers]

    scenario = LoadScenario(
        data=generate_scenario_yaml(blocks, rng)
    )

    station = MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
        package_xmls=["assets/models/package.xml"],
    )

    builder = DiagramBuilder()
    builder.AddSystem(station)
    pc_systems = AddPointClouds(scenario=scenario, station=station, builder=builder)

    builder.ExportOutput(pc_systems["camera0"].get_output_port(), "camera0_point_cloud")
    builder.ExportOutput(pc_systems["camera1"].get_output_port(), "camera1_point_cloud")
    builder.ExportOutput(pc_systems["camera2"].get_output_port(), "camera2_point_cloud")

    plant = station.GetSubsystemByName("plant")

    # diffIK setup
    controller = builder.AddSystem(
        PseudoInverseDiffIK(plant, iiwa_model_name='iiwa', ee_body_name="body")
    )
    integrator = builder.AddSystem(Integrator(7))
    commander = builder.AddSystem(TrajectoryCommander(dt_fd=1e-3))

    # feed measured q into controller
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        controller.GetInputPort("q_measured")
    )
    # commander outputs desired end-effector spatial velocity
    builder.Connect(
        commander.GetOutputPort("V_WG_des"),
        controller.GetInputPort("V_WG_des")
    )
    # controller qdot -> integrator -> iiwa.position command
    builder.Connect(controller.GetOutputPort("qdot_cmd"), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
    # gripper commands
    builder.Connect(commander.GetOutputPort("wsg_position"), station.GetInputPort("wsg.position"))
    connect_if_present(builder, station, "wsg.force_limit", np.array([40.0]))  # must be > 0

    # optional safety zeros if ports exist
    connect_if_present(builder, station, "iiwa.velocity", np.zeros(7))
    connect_if_present(builder, station, "iiwa.feedforward_torque", np.zeros(7))

    return builder, station, meshcat, pc_systems, commander, integrator

def generate_model_point_cloud(block_idx: int):
    N_SAMPLE_POINTS = 1500
    block_mesh_path = Path('assets') / f"block{block_idx}.obj"
    block_mesh = trimesh.load(block_mesh_path.as_posix(), force="mesh")
    sampled_pts = np.asarray(block_mesh.sample(N_SAMPLE_POINTS))
    block_model_cloud = PointCloud(sampled_pts.T)
    print(f"Loaded model mesh from {block_mesh_path}, sampled {sampled_pts.shape[0]} points")
    return block_model_cloud

def pick_block(estimated_X_WB, plant, plant_context, diagram, diagram_context, meshcat, pc_systems, place_xy: np.ndarray, place_z: float):
    """
    Pipeline for pick + placing one block. 
    System currently half-cheats for perception (identifying blocks using their true pose).
    TODO: ^ identify blocks using a different method (e.g., clustering)
    """
    # pick one block
    # block_name = f"block{block_idx}"
    # block_inst = plant.GetModelInstanceByName(block_name)
    # block_body = plant.GetBodyByName(f"{block_name}_link", block_inst)
    # X_WB_true = plant.EvalBodyPoseInWorld(plant_context, block_body)

    # perception crop + merge
    # block_cloud = preprocess_block_cloud(
    #     diagram, diagram_context, X_WB_true, pc_systems
    # )

    # visualize crop box + cloud
    # meshcat.SetObject("block_cloud", block_cloud, point_size=0.01, rgba=Rgba(1, 0, 0))

    # estimate pose from perception
    # X_WB_hat = estimate_pose_pca(block_cloud)
    X_WB_hat = estimated_X_WB[0]
    # extents_hat, _, _ = estimate_extents_along_axes(block_cloud, X_WB_hat)
    extents_hat = 0.06, estimated_X_WB[1], 0.06
    print("Estimated extents:", extents_hat)

    # compare to truth (sanity check)
    # err = X_WB_hat.inverse().multiply(X_WB_true)
    # xxprint("Pose error rpy:", RollPitchYaw(err.rotation()).vector(), "xyz:", err.translation())

    # design grasp
    X_WG_pre, X_WG_pick = design_top_down_grasp(X_WB_hat, extents_hat, ee_approach_axis="y", ee_close_axis="x")
    
    AddMeshcatTriad(meshcat, "X_WB_hat", X_PT=X_WB_hat, length=0.15)
    AddMeshcatTriad(meshcat, "X_WG_pre", X_PT=X_WG_pre, length=0.15)
    AddMeshcatTriad(meshcat, "X_WG", X_PT=X_WG_pick, length=0.15)

    # initial gripper pose from plant
    ee_body = plant.GetBodyByName("body")  # wsg body
    X_WG_initial = plant.EvalBodyPoseInWorld(plant_context, ee_body)

    # trajectories
    # pose_traj, V_source, use_derivative_source, wsg_source, wsg_traj = make_pick_trajectories(
    #     X_WG_initial, X_WG_pre, X_WG_pick
    # )
    pose_traj, wsg_source, wsg_traj = make_pick_and_place_trajectories(
        X_WG_initial=X_WG_initial, 
        X_WG_pre_pick=X_WG_pre,
        X_WG_pick=X_WG_pick,
        place_xy=place_xy,
        place_z=place_z,
    )
    return pose_traj, wsg_traj

# ENTRY POINT

def main():
    builder, station, meshcat, pc_systems, commander, integrator = generate_setup()
    plant = station.GetSubsystemByName("plant")

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)


    # publish once so camera clouds exist
    diagram.ForcedPublish(diagram_context)

    input("enter to continue")

    # computing platofrm
    platform_inst = plant.GetModelInstanceByName("platform")
    platform_body = plant.GetBodyByName("platform_link", platform_inst)
    X_WPlat = plant.EvalBodyPoseInWorld(plant_context, platform_body)
    p_WPlat = X_WPlat.translation()

    # place at platform center, at the first layer height
    place_xy = p_WPlat[:2]
    platform_half_h = 0.05 / 2.0  # from create_block_sdf for platform
    block_h = BLOCK_HEIGHT

    # initialize integrator state to current measured q
    station_context = diagram.GetMutableSubsystemContext(station, diagram_context)
    q_meas = station.GetOutputPort("iiwa.position_measured").Eval(station_context)

    integrator_context = diagram.GetMutableSubsystemContext(integrator, diagram_context)
    integrator.set_integral_value(integrator_context, q_meas)


    simulator = Simulator(diagram, diagram_context)

    meshcat.StartRecording()
    simulator.AdvanceTo(5) # allow time for blocks to drop before getting point clouds
    current_time = simulator.get_context().get_time()

    point_cloud = get_point_cloud_from_cameras(diagram, diagram_context)
    est_X_WBs_by_length = perceive(point_cloud, meshcat)

    for stack_level in range(len(est_X_WBs_by_length)):
        print("platform_half_h = ", platform_half_h)
        print("block_h = ", block_h)
        place_z = platform_half_h + (stack_level + 1 + 0.5) * block_h

        # plan using perception
        pose_traj, wsg_traj = pick_block(
            # stack_level + 1,
            est_X_WBs_by_length[stack_level], # est_X_WBs_by_length,
            plant, plant_context,
            diagram, diagram_context,
            meshcat, pc_systems,
            place_xy, place_z
        )
        print("place_z: ", place_z)

        # set trajectories into commander
        commander.set_trajectories(pose_traj, wsg_traj)
        commander.set_time_offset(current_time)

        # run just long enough to finish this block's motion
        T_block = pose_traj.end_time()
        block_end_global = current_time + T_block
        simulator.AdvanceTo(block_end_global)
        current_time = block_end_global

    meshcat.StopRecording()
    meshcat.PublishRecording()

if __name__ == "__main__":
    main()
    input("Press Enter to shut down Meshcat and exit...\n") # this is to keep meshcat alive