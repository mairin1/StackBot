# Code adapted from Problem Set 5
import numpy as np

from pydrake.all import DiagramBuilder, RigidTransform
from manipulation.exercises.trajectories.rrt_planner.robot import ConfigurationSpace, Range
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import RRT, TreeNode
from manipulation.station import LoadScenario, MakeHardwareStation

from constants import *
from scene_utils import *

class ManipulationStationSim:
    def __init__(
        self,
        scenario_file: str | None = None,
        q_iiwa: tuple | None = None,
        gripper_setpoint: float = 0.1,
        meshcat=None,
        block_poses: dict[str, "RigidTransform"] | None = None,
    ) -> None:

        self.scenario = None
        self.station = None
        self.plant = None
        self.scene_graph = None
        self.query_output_port = None
        self.diagram = None

        self.meshcat = meshcat

        # contexts
        self.context_diagram = None
        self.context_station = None
        self.context_scene_graph = None
        self.context_plant = None

        # mark initial configuration
        self.q0 = None

        self.okay_collisions = None
        self.gripper_setpoint = gripper_setpoint

        if scenario_file is not None:
            self.choose_sim(scenario_file, q_iiwa, gripper_setpoint, block_poses)

    def choose_sim(
        self,
        scenario_file: str,
        q_iiwa: tuple | None = None,
        gripper_setpoint: float = 0.1,
        block_poses: dict[str, "RigidTransform"] | None = None,
    ) -> None:
        """build the world for planning with RRT"""

        self.clear_meshcat()

        self.scenario = LoadScenario(filename=scenario_file)
        builder = DiagramBuilder()
        self.station = builder.AddSystem(
            MakeHardwareStation(self.scenario, meshcat=self.meshcat, package_xmls=["assets/models/package.xml"],)
        )

        self.plant = self.station.GetSubsystemByName("plant") #type:ignore

        self.scene_graph = self.station.GetSubsystemByName("scene_graph") #type:ignore

        # scene graph query output port.
        self.query_output_port = self.scene_graph.GetOutputPort("query")

        self.diagram = builder.Build()

        # contexts
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_station = self.diagram.GetSubsystemContext(
            self.station, self.context_diagram
        )
        self.station.GetInputPort("iiwa.position").FixValue(
            self.context_station, np.zeros(7) #type:ignore
        )
        self.station.GetInputPort("wsg.position").FixValue(self.context_station, [0.1]) #type:ignore
        self.context_scene_graph = self.station.GetSubsystemContext(
            self.scene_graph, self.context_station
        )
        self.context_plant = self.station.GetMutableSubsystemContext(
            self.plant, self.context_station
        )

        # same randomized block poses TODO can remove blocks entirely tbh
        if block_poses is not None:
            for block_name, X_WB in block_poses.items():
                model_inst = self.plant.GetModelInstanceByName(block_name)
                body = self.plant.GetBodyByName(f"{block_name}_link", model_inst)
                self.plant.SetFreeBodyPose(self.context_plant, body, X_WB)
        self.diagram.ForcedPublish(self.context_diagram)

        # mark initial configuration
        self.gripper_setpoint = gripper_setpoint
        if q_iiwa is None:
            self.q0 = self.plant.GetPositions(
                self.context_plant, self.plant.GetModelInstanceByName("iiwa")
            )
        else:
            self.q0 = q_iiwa
            self.SetStationConfiguration(q_iiwa, gripper_setpoint)

        self.DrawStation(self.q0, 0.1)
        # baseline allowed contacts, *excluding* any involving blocks
        self.okay_collisions = len(self._penetrations_excluding_blocks())
        print("Baseline (non-block) collision count =", self.okay_collisions)

    def clear_meshcat(self) -> None:
        if self.meshcat is not None:
            self.meshcat.Delete()

    def SetStationConfiguration(self, q_iiwa: tuple, gripper_setpoint: float) -> None:
        """
        :param q_iiwa: (7,) tuple, joint angle of robots in radian.
        :param gripper_setpoint: float, gripper opening distance in meters.
        :return:
        """
        self.plant.SetPositions(
            self.context_plant,
            self.plant.GetModelInstanceByName("iiwa"),
            q_iiwa,
        )
        self.plant.SetPositions(
            self.context_plant,
            self.plant.GetModelInstanceByName("wsg"),
            [-gripper_setpoint / 2, gripper_setpoint / 2],
        )

    def DrawStation(self, q_iiwa: tuple, gripper_setpoint: float = 0.1) -> None:
        self.SetStationConfiguration(q_iiwa, gripper_setpoint)
        self.diagram.ForcedPublish(self.context_diagram)

    def _penetrations_excluding_blocks(self, verbose: bool = False):
        """
        Returns a list of penetration pairs, excluding any pair that
        involves a block body (body name contains 'block', case-insensitive).
        """
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        all_pairs = query_object.ComputePointPairPenetration()

        inspector = self.scene_graph.model_inspector()
        filtered = []

        if verbose:
            print("=== Raw penetration pairs ===")

        for pair in all_pairs:
            keep = True
            bodies = []

            for geom_id in (pair.id_A, pair.id_B):
                frame_id = inspector.GetFrameId(geom_id)
                body = self.plant.GetBodyFromFrameId(frame_id)
                body_name = body.name()
                model_instance = body.model_instance()
                model_name = self.plant.GetModelInstanceName(model_instance)

                bodies.append((body_name, model_name))

                # treat ANY body whose name contains 'block' as a block
                if "block" in body_name.lower() or "block" in model_name.lower():
                    keep = False

            if verbose:
                print(f"  pair: {bodies}, keep={keep}")

            if keep:
                filtered.append(pair)

        if verbose:
            print("=== End penetration list ===")
            print(f"Filtered (non-block) count = {len(filtered)}")

        return filtered


    def ExistsCollision(self, q_iiwa: tuple, gripper_setpoint: float) -> bool:
        """
        Checks for an unwanted collision for a given robot configuration
        (q_iiwa) and gripper setpoint (gripper_setpoint)

        Args:
            q_iiwa: given robot configuration
            gripper_setpoint: gripper width
        Returns:
            bool: True if an unwnted collision exists, False otherwise
        """

        self.SetStationConfiguration(q_iiwa, gripper_setpoint)
        filtered_pairs = self._penetrations_excluding_blocks()

        # any *new* non-block collision beyond the baseline is considered bad
        return len(filtered_pairs) > self.okay_collisions #type:ignore


class RRT_Connect_tools:
    def __init__(
        self,
        sim: ManipulationStationSim,
        start: tuple,
        goal: tuple,
        gripper_setpoint: float = 0.2
    ) -> None:

        self.sim = sim
        self.start = start
        self.goal = goal
        self.gripper_setpoint = float(gripper_setpoint)

        nq = 7
        joint_limits = np.zeros((nq, 2))
        for i in range(nq):
            joint = sim.plant.GetJointByName("iiwa_joint_%i" % (i + 1))
            joint_limits[i, 0] = joint.position_lower_limits()[0]
            joint_limits[i, 1] = joint.position_upper_limits()[0]

        range_list = []
        for joint_limit in joint_limits:
            range_list.append(Range(joint_limit[0], joint_limit[1]))

        def l2_distance(q: tuple):
            sum = 0
            for q_i in q:
                sum += q_i**2
            return np.sqrt(sum)

        max_steps = nq * [np.pi / 180 * 1.5]  # two degrees
        self.cspace = ConfigurationSpace(range_list, l2_distance, max_steps)
        self.rrt_tree_start = RRT(TreeNode(start), self.cspace)
        self.rrt_tree_goal = RRT(TreeNode(goal), self.cspace)

    def find_nearest_node_in_RRT_graph(self, q_sample: tuple) -> TreeNode:
        """Return nearest node to q_sample in a single-tree context (expects self.rrt_tree)."""
        nearest_node = self.rrt_tree.nearest(q_sample)
        return nearest_node

    def sample_node_in_configuration_space(self) -> tuple:
        """Sample a random valid configuration from the c-space."""
        q_sample = self.cspace.sample()
        return q_sample

    def calc_intermediate_qs_wo_collision(
        self, start: tuple, end: tuple, gripper_setpoint: float | None
    ) -> list[tuple]:
        """
        Checks if the path from start to end collides with any obstacles.

        Args:
            start: tuple of floats - tuple describing robot's start
                configuration
            end: tuple of floats - tuple describing robot's end configuration

        Returns:
            list of tuples along the path that are not in collision.
        """
        if gripper_setpoint is None:
            gripper_setpoint = self.gripper_setpoint
        path = self.cspace.path(start, end)
        # print("[direct] raw cspace.path len:", len(path))
        # if len(path) > 0:
        #     print("[direct] first config:", path[0])
        #     print("[direct] last  config:", path[-1])
        safe_path = []
        for configuration in path:
            if self.sim.ExistsCollision(np.array(configuration), gripper_setpoint): #type: ignore
                return safe_path
            safe_path.append(configuration)
        return safe_path

    def node_reaches_goal(self, q_step: tuple, tol: float = 1e-2) -> bool:
        """Check if q_step is within tol of the goal in c-space distance."""
        return self.cspace.distance(q_step, self.goal) <= tol

    def backup_path_from_node(self, node: TreeNode) -> list[tuple]:
        """Reconstruct path from tree root to the given node (inclusive)."""
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path

    def extend_once(self, tree: RRT, q_target: tuple) -> TreeNode | None:
        """Extend tree by one step toward q_target (returns new node or None if blocked)."""
        q_near_node = tree.nearest(q_target)
        edge = self.calc_intermediate_qs_wo_collision(q_near_node.value, q_target, self.gripper_setpoint)
        if len(edge) <= 1:
            return None
        q_step = edge[1]
        new_node = tree.add_configuration(q_near_node, q_step)

        return new_node

    def connect_greedy(
        self, tree: RRT, q_target: tuple, eps: float = 1e-2
    ) -> tuple[TreeNode | None, bool]:
        """
        Greedily add as many collision-free segments as possible toward q_target.

        Returns:
            (last_node, complete): last_node reached; complete=True if within eps.
        """
        near_node = tree.nearest(q_target)
        q_near_node = near_node.value
        path = self.calc_intermediate_qs_wo_collision(q_near_node, q_target, self.gripper_setpoint)
        if len(path) > 1:
            last_node = near_node
            for j in range(1, len(path)):
                last_node = tree.add_configuration(last_node, path[j])

            return last_node, (self.cspace.distance(last_node.value, q_target) < eps)

        return (None, False)

    @staticmethod
    def concat_paths(path_a: list[tuple], path_b: list[tuple]) -> list[tuple]:
        """Concatenate two paths, de-duplicating the shared joint at the seam."""
        if path_a and path_b and path_a[-1] == path_b[0]:
            return path_a + path_b[1:]
        return path_a + path_b
    
def rrt_connect_planning(
    sim,
    q_start: tuple,
    q_goal: tuple,
    max_iterations: int = 2000,
    eps: float = 1e-2,
    gripper_setpoint: float = 0.2
):
    """
    Run RRT-Connect using the RRT_Connect_tools interface.

    Args:
        sim: ManipulationStationSim (must implement ExistsCollision)
        q_start: start configuration (tuple of 7 floats)
        q_goal: goal configuration (tuple of 7 floats)
        max_iterations: maximum number of tree expansion attempts
        eps: tolerance for declaring the trees "connected"

    Returns:
        (path, iterations)
            path: list of joint tuples from q_start to q_goal, or None
            iterations: number of iterations executed
    """

    tools = RRT_Connect_tools(sim, start=q_start, goal=q_goal, gripper_setpoint=gripper_setpoint)
    T_start = tools.rrt_tree_start
    T_goal = tools.rrt_tree_goal

    active_is_start = True

    for it in range(max_iterations):

        # 1. sample a random configuration
        q_rand = tools.sample_node_in_configuration_space()

        # 2. pick which tree to grow this iteration
        T_active = T_start if active_is_start else T_goal
        T_other  = T_goal  if active_is_start else T_start

        # 3. extend the active tree toward the sample
        node_a = tools.extend_once(T_active, q_rand)
        if node_a is None:
            active_is_start = not active_is_start
            continue

        # 4. try to greedily connect the other tree to node_a
        node_b, complete = tools.connect_greedy(T_other, node_a.value, eps)

        # 5. if the two trees have connected, reconstruct path
        if complete and node_b is not None:

            # paths from each root to the connection point
            path_a = tools.backup_path_from_node(node_a)
            path_b = tools.backup_path_from_node(node_b)

            # if the goal tree was the active tree, reverse concatenation direction
            if not active_is_start:
                path_a, path_b = path_b, path_a

            # remove duplicate seam if any and concatenate
            full_path = RRT_Connect_tools.concat_paths(path_a, list(reversed(path_b)))

            return full_path, it

        # alternate trees
        active_is_start = not active_is_start

    # no path found
    return None, max_iterations
