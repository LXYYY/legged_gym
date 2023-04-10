from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AirHockeyCfg(LeggedRobotCfg):
    n_agents = 1

    def __init__(self):
        super().__init__()

    class env(LeggedRobotCfg.env):
        num_envs = 3
        num_observations = 12
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 11
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class asset(LeggedRobotCfg.asset):
        # xml's color not working somehow
        colors = {
            'puck': [1, 0, 0],
            'rim': [0, 0, 1],
            'table_surface': [0, 1, 0],
            # gold
            'planar_robot_1/body_ee': [0.8, 0.8, 0.2],
        }
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        disable_gravity = False
        solref = [0.02, 0.3]
        terminate_after_contacts_on = ['rim']

    class sim(LeggedRobotCfg.sim):
        dt = 0.001

    class control(LeggedRobotCfg.control):
        decimation = 1
        control_joint_idx = {
            'planar_robot_1/joint_1': 0,
            'planar_robot_1/joint_2': 1,
            'planar_robot_1/joint_3': 2
        }
        no_sim_body = {
            'planar_robot_1/base',
            'planar_robot_1/body_1',
            'planar_robot_1/body_2',
            'planar_robot_1/body_3',
            'planar_robot_1/body_ee',
            'planar_robot_1/body_hand'
        }
        control_type = 'P'
        stiffness = {'planar_robot_1/joint_1': 960, 'planar_robot_1/joint_2': 480, 'planar_robot_1/joint_3': 240}
        damping = {'planar_robot_1/joint_1': 60, 'planar_robot_1/joint_2': 20, 'planar_robot_1/joint_3': 4}
        # Frames chain: world -> env(?) -> air_hockey -> robot_base -> robot_ee/puck
        robot_base_body = 'planar_robot_1/base'
        actor_body = 'air_hockey'
        ee_body = 'planar_robot_1/body_ee'

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
        pos = [0, 0, 0]

    class viewer(LeggedRobotCfg.viewer):
        #     ref_env = 0
        pos = [3, 0, 12]  # [m]
        lookat = [0, 0, 1]  # [m]

    class rewards:
        class scales:
            termination = 0.0
            ee_pos = -0.1
            ee_vel = -0.1
            jerk = -0.1
            collision = 0
            termination = -10000

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.  # forces above this value are penalized

        only_positive_rewards = False
        max_puck_vel = 1.
        min_puck_ee_dist = 0.1
        max_vel_trunc_dist = 0.5

        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.  # forces above this value are penalized

class AirHockeyCfgPPO(LeggedRobotCfgPPO):
    num_actions = 6
