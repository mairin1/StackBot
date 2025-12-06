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

    generate_blocks()
    create_camera_directives()
    generate_directives_yaml(NUM_BLOCKS)
    generate_scenario_yaml()

    scenario = LoadScenario(
        filename="scenarios/bimanual_IIWA14_stackbot_assets_and_cameras.scenario.yaml"
    )

    station = MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
        package_xmls=["assets/models/package.xml"],
    )

    builder = DiagramBuilder()
    builder.AddSystem(station)
    pc_systems = AddPointClouds(scenario=scenario, station=station, builder=builder)

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

def pick_block(block_idx: int, plant, plant_context, diagram, diagram_context, meshcat, pc_systems):
    """
    Pipeline for pick + placing one block. 
    System currently half-cheats for perception (identifying blocks using their true pose).
    TODO: ^ identify blocks using a different method (e.g., clustering)
    """
    # pick one block
    block_name = f"block{block_idx}"
    block_inst = plant.GetModelInstanceByName(block_name)
    block_body = plant.GetBodyByName(f"{block_name}_link", block_inst)
    X_WB_true = plant.EvalBodyPoseInWorld(plant_context, block_body)

    # perception crop + merge
    block_cloud = preprocess_block_cloud(
        diagram, diagram_context, X_WB_true, pc_systems
    )

    # visualize crop box + cloud
    meshcat.SetObject("block_cloud", block_cloud, point_size=0.01, rgba=Rgba(1, 0, 0))

    # estimate pose from perception
    X_WB_hat = estimate_pose_pca(block_cloud)
    extents_hat, _, _ = estimate_extents_along_axes(block_cloud, X_WB_hat)
    print("Estimated extents:", extents_hat)

    # compare to truth (sanity check)
    err = X_WB_hat.inverse().multiply(X_WB_true)
    print("Pose error rpy:", RollPitchYaw(err.rotation()).vector(), "xyz:", err.translation())

    # design grasp
    X_WG_pre, X_WG_pick = design_top_down_grasp(X_WB_hat, extents_hat, ee_approach_axis="y", ee_close_axis="x")

    AddMeshcatTriad(meshcat, "X_WB_hat", X_PT=X_WB_hat, length=0.15)
    AddMeshcatTriad(meshcat, "X_WG_pre", X_PT=X_WG_pre, length=0.15)
    AddMeshcatTriad(meshcat, "X_WG", X_PT=X_WG_pick, length=0.15)

    # initial gripper pose from plant
    ee_body = plant.GetBodyByName("body")  # wsg body
    X_WG_initial = plant.EvalBodyPoseInWorld(plant_context, ee_body)

    # trajectories
    pose_traj, V_source, use_derivative_source, wsg_source, wsg_traj = make_pick_trajectories(
        X_WG_initial, X_WG_pre, X_WG_pick
    )
    return pose_traj, wsg_traj

# ENTRY POINT

def main():
    builder, station, meshcat, pc_systems, commander, integrator = generate_setup()
    plant = station.GetSubsystemByName("plant")

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    blocks = [f"block{i}" for i in range(1, NUM_BLOCKS + 1)]
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    randomize_blocks_near_kuka(blocks, plant, plant_context)

    # publish once so camera clouds exist
    diagram.ForcedPublish(diagram_context)

    input("enter to continue")

    # plan using perception
    pose_traj, wsg_traj = pick_block(
        1, plant, plant_context, diagram, diagram_context, meshcat, pc_systems
    )

    print("wsg_traj: ", wsg_traj)

    # set trajectories into commander
    commander.set_trajectories(pose_traj, wsg_traj)

    # initialize integrator state to current measured q
    station_context = diagram.GetMutableSubsystemContext(station, diagram_context)
    q_meas = station.GetOutputPort("iiwa.position_measured").Eval(station_context)

    integrator_context = diagram.GetMutableSubsystemContext(integrator, diagram_context)
    integrator.set_integral_value(integrator_context, q_meas)

    simulator = Simulator(diagram, diagram_context)

    meshcat.StartRecording()
    if running_as_notebook:
        simulator.set_target_realtime_rate(1.0)

    sim_end = commander.end_time()
    if sim_end <= 0:
        raise RuntimeError("Commander has no trajectory (end_time <= 0).")

    simulator.AdvanceTo(sim_end)
    meshcat.StopRecording()
    meshcat.PublishRecording()

if __name__ == "__main__":
    main()
    input("Press Enter to shut down Meshcat and exit...\n") # this is to keep meshcat alive