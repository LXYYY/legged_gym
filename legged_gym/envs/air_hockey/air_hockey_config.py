from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AirHockeyCfg(LeggedRobotCfg):
    n_agents = 1

    def __init__(self):
        super().__init__()

    class env(LeggedRobotCfg.env):
        num_envs = 1000
        num_observations = 12
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 11
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 3  # episode length in seconds

        hierarchical = True

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
        penalize_contacts_on = ['planar_robot_1/body_ee']

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
            ee_pos = -100
            # final_ee_vel = 10
            # jerk = -100
            # collision = -1e4
            # termination = -0
            #
            # torques = -5e-7
            # dof_vel = -5e-7
            # dof_acc = -2.5e-7

        class mid_scales:
            ee_pos_subgoal = -1
            # ee_vel_subgoal = -1

        class low_scales:
            dof_pos_subgoal = -60
            low_termination = 1000
            torques = -5e-6
            dof_vel_subgoal = -5
            dof_vel = -5e-3
            dof_acc = -2.5e-7
            dof_pos_limits = -1e4
            dof_vel_limits = -1e4
            torque_limits = -1e4

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.  # forces above this value are penalized

        only_positive_rewards = False
        max_puck_vel = 0.4
        min_puck_ee_dist = 0.01
        max_vel_trunc_dist = 0.5
        min_dof_pos_done = 0.002  # rad >~ 0.1 deg

        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.  # forces above this value are penalized


class AirHockeyCfgPPO(LeggedRobotCfgPPO):
    num_actions = 6

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 30

    class policy:
        class high(LeggedRobotCfgPPO.policy):
            num_actions = 4  # x,y,vel_x,vel_y
            num_obs = 12  # num_obs+t
            num_steps = 1
            num_steps_per_env = 10
            actor_hidden_dims = [64, 64]
            critic_hidden_dims = [64, 64]
            obs_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


        class mid(LeggedRobotCfgPPO.policy):
            num_actions = 6  # q, qd for 3 joints
            num_obs = 10  # 6+high_actions q, qd for 3 joints
            num_steps = 20
            num_steps_per_env = 50
            actor_hidden_dims = [128, 64, 32]
            critic_hidden_dims = [128, 64, 32]
            obs_idx = [6, 7, 8, 9, 10, 11]
            init_noise_std = 0.05

        class low(LeggedRobotCfgPPO.policy):
            num_actions = 6  # q, qd for 3 joints
            num_obs = 12  # 6+mid_actions q, qd for 3 joints
            num_steps_per_env = 1000
            actor_hidden_dims = [64, 32]
            critic_hidden_dims = [64, 32]
            obs_idx = [6, 7, 8, 9, 10, 11]
            init_noise_std = 0.5
