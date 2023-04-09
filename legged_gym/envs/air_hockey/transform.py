from isaacgym import gymapi, gymtorch, gymutil
import numpy as np


def T_mat_from_body_state(body_state: gymapi.RigidBodyState):
    T = np.eye(4)
    quat = body_state[3:7]
    # quat to matrix
    T[:3, :3] = gymutil.quat_to_mat(quat)
    T[:3, 3] = body_state.pose.p
    return T, quat
