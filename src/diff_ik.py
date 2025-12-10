import numpy as np
from pydrake.all import(
    BasicVector,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    RigidTransform
)

class PseudoInverseDiffIK(LeafSystem):
    """
    Inputs:
      - q_measured (7)
      - V_WG_des (6) spatial velocity of end-effector in world
    Output:
      - qdot_cmd (7)
    """
    def __init__(self, plant, iiwa_model_name="iiwa", ee_body_name="body",
                 damping=1e-3, v_limit=1.5):
        super().__init__()
        self._plant = plant
        self._context = plant.CreateDefaultContext()

        self._iiwa = plant.GetModelInstanceByName(iiwa_model_name)
        self._ee_body = plant.GetBodyByName(ee_body_name)
        self._W = plant.world_frame()
        self._E = self._ee_body.body_frame()

        self._nv_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self._nv = plant.num_velocities(self._iiwa)

        self._damping = damping
        self._v_limit = v_limit

        self.DeclareVectorInputPort("q_measured", BasicVector(self._nv))
        self.DeclareVectorInputPort("V_WG_des", BasicVector(6))
        self.DeclareVectorOutputPort("qdot_cmd", BasicVector(self._nv), self._calc)

    def _calc(self, context, output):
        q = self.get_input_port(0).Eval(context)
        V = self.get_input_port(1).Eval(context) # [wx, wy, wz, vx, vy, vz] in world

        # update plant context
        self._plant.SetPositions(self._context, self._iiwa, q)

        p_BoBp_B = np.zeros((3, 1))

        J = self._plant.CalcJacobianSpatialVelocity(
            self._context,
            JacobianWrtVariable.kQDot,
            self._E, # frame_B  (the EE frame)
            p_BoBp_B, 
            self._W, # frame_A  (measured in world)
            self._W, # frame_E  (expressed in world)
        )  # 6 x plant.num_velocities()

        # extract iiwa block
        J_iiwa = J[:, self._nv_start:self._nv_start + self._nv]  # 6x7

        # damped least squares: qdot = J^T (J J^T + λI)^-1 V
        lam = self._damping
        A = J_iiwa @ J_iiwa.T + lam * np.eye(6)
        qdot = J_iiwa.T @ np.linalg.solve(A, V)

        # clip
        qdot = np.clip(qdot, -self._v_limit, self._v_limit)
        output.SetFromVector(qdot)

  
def refine_with_diffik(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    q_init: np.ndarray,
    iiwa_model_name: str = "iiwa",
    ee_body_name: str = "body",
    max_iters: int = 50,
    dt: float = 0.05,
    k_pos: float = 2.0,
    k_rot: float = 2.0,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-3,
) -> np.ndarray:
    """
    Local diff-IK refinement around q_init to better match X_WG_target.
    Uses a simple damped pseudo-inverse Jacobian step in a loop.

    Returns:
        q_refined (7,) as numpy array.
    """
    # create a private context for kinematics
    context = plant.CreateDefaultContext()

    iiwa = plant.GetModelInstanceByName(iiwa_model_name)
    ee_body = plant.GetBodyByName(ee_body_name)
    W = plant.world_frame()
    E = ee_body.body_frame()

    q = np.array(q_init, dtype=float).copy()
    q_lower = plant.GetPositionLowerLimits()[:7]
    q_upper = plant.GetPositionUpperLimits()[:7]

    def pose_error(q_vec):
        plant.SetPositions(context, iiwa, q_vec)
        X_WG_curr = plant.CalcRelativeTransform(context, W, E)
        R_curr = X_WG_curr.rotation().matrix()
        p_curr = X_WG_curr.translation()

        R_des = X_WG_target.rotation().matrix()
        p_des = X_WG_target.translation()

        # position error
        e_p = p_des - p_curr  # 3

        # orientation error via axis-angle of R_err = R_des * R_curr^T
        R_err = R_des @ R_curr.T
        # numerical safety
        cos_theta = (np.trace(R_err) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-9:
            e_w = np.zeros(3)
        else:
            # axis from skew-symmetric part
            wx = R_err[2,1] - R_err[1,2]
            wy = R_err[0,2] - R_err[2,0]
            wz = R_err[1,0] - R_err[0,1]
            axis = np.array([wx, wy, wz]) / (2.0 * np.sin(theta))
            e_w = theta * axis  # 3

        return e_p, e_w

    for _ in range(max_iters):
        e_p, e_w = pose_error(q)
        print("pick pose residual:", np.linalg.norm(e_p), np.linalg.norm(e_w))

        if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_w) < tol_rot:
            print("close enough")
            break

        # desired twist in world frame
        v_des = k_pos * e_p
        w_des = k_rot * e_w
        V_des = np.hstack([w_des, v_des])  # (6,)

        # Jacobian J (6 x nv), then extract iiwa block (7 joints)
        p_BoBp_B = np.zeros((3, 1))
        J_full = plant.CalcJacobianSpatialVelocity(
            context,
            JacobianWrtVariable.kQDot,
            E,
            p_BoBp_B,
            W,
            W,
        )

        # assume iiwa joints are first 7 velocities
        J = J_full[:, :7]

        lam = 1e-3
        A = J @ J.T + lam * np.eye(6)
        qdot = J.T @ np.linalg.solve(A, V_des)

        # simple Euler step
        q = q + dt * qdot
        q = np.clip(q, q_lower, q_upper)

    return q
