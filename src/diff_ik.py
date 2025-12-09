import numpy as np
from pydrake.all import(
    BasicVector,
    JacobianWrtVariable,
    LeafSystem
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