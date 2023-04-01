from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AirHockeyCfg(LeggedRobotCfg):
    n_agents = 1

    def __init__(self):
        super().__init__()

    class env(LeggedRobotCfg.env):
        num_envs = 1
        num_observations = 12
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 10
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class asset_table(LeggedRobotCfg.asset):
        file = ''
        name = 'table'

    class asset_robot(LeggedRobotCfg.asset):
        file = ''
        name = 'robot'

    class sim(LeggedRobotCfg.sim):
        dt = 0.001

    class control(LeggedRobotCfg.control):
        decimation = 20

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            'planar_robot_1/joint_1': -1.15570723,
            'planar_robot_1/joint_2': 1.30024401,
            'planar_robot_1/joint_3': 1.44280414
        }
        random_joints = {
            'pack_x',
            'pack_y',
            'pack_yaw'
        }
        control_joint_idx = {
            'planar_robot_1/joint_1': 0,
            'planar_robot_1/joint_2': 1,
            'planar_robot_1/joint_3': 2
        }
        puck_pos = {
            'puck_x': 0,
            'puck_y': 1,
            'puck_yaw': 2
        }
        puck_vel = {
            'puck_x': 0,
            'puck_y': 1,
            'puck_yaw': 2
        }
        pos = [0, 0, 0.02]


class AirHockeyCfgPPO(LeggedRobotCfgPPO):
    pass
