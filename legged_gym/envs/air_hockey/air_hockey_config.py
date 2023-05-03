from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AirHockeyCfg(LeggedRobotCfg):
    n_agents = 1

    def __init__(self):
        super().__init__()

    class env(LeggedRobotCfg.env):
        num_envs = 10000
        num_observations = 15  # original 12 + step + goal
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 11
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = False  # send time out information to the algorithm
        episode_length_s = 2  # episode length in seconds

        goal_x = 2.484
        goal_width = 0.3
        contact_force_threshold = 0.1

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
        # penalize_contacts_on = ['planar_robot_1/body_ee']

    class sim(LeggedRobotCfg.sim):
        dt = 0.005

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
        control_type = 'T'
        stiffness = {'planar_robot_1/joint_1': 100, 'planar_robot_1/joint_2': 100, 'planar_robot_1/joint_3': 100}
        damping = {'planar_robot_1/joint_1': 10, 'planar_robot_1/joint_2': 10, 'planar_robot_1/joint_3': 10}
        # Frames chain: world -> env(?) -> air_hockey -> robot_base -> robot_ee/puck
        robot_base_body = 'planar_robot_1/base'
        actor_body = 'air_hockey'
        ee_body = 'planar_robot_1/body_ee'

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            puck_pos = 0.01
            puck_vel = 0.01
            joint_pos = 0.01
            joint_vel = 0.01
            episode_length = 0

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            # 'planar_robot_1/joint_1': -1.15570723,
            # 'planar_robot_1/joint_2': 1.7,
            # 'planar_robot_1/joint_3': 0
            'planar_robot_1/joint_1': -1.15570723,
            'planar_robot_1/joint_2': 1.30024401,
            'planar_robot_1/joint_3': 1.44280414/2

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
        pos = [3, 0, 2]  # [m]
        lookat = [0, 0, 1]  # [m]

    class rewards:
        class scales:
            time_utl_success = -1
            high_termination = -50
            # no_success_no_fail = -100000
            ee_pos = -50
            puck_outside_table = -1000
            ee_collision=-100
            ee_outside_table=-2000
            # ee_pd_limit=-20
            # hit_puck = 1
            # puck_x = 10
            # puck_y = 10
            torques = -1e-3
            jerk = -1e-3

            # ee_collision=-100000
            
            # ee_outside_table=-10000
            # ee_collision=-100
            # ee_outside_table=-100
            # ee_puck_contact = 1000
            # final_ee_vel = 10
            # collision = -1e4
            # termination = -0
            #
            # torques = -5e-7
            # dof_vel = -5e-7
            # dof_acc = -2.5e-9

        class mid_scales:
            # ee_pos_subgoal = -10
            mid_termination = 50
            dof_qd_limit=-10
            ee_outside_table=-50
            # torques = -5e-4
            # jerk = -5e-7
            # ee_vel_subgoal = -0.2
            # puck_outside_table = -10

        class low_scales:
            dof_pos_subgoal = -10
            # dof_vel_subgoal = -1
            low_termination = 5
            ee_outside_table=-10000
            # torques = -5e-7
            # dof_acc = -1e-8
            # dof_pos_limits = -1e4
            # dof_vel_limits = -1e2
            torque_limits = -1
            torques = -5e-4
            jerk = -1e-7
            # puck_outside_table = -10000
            # ee_outside_table=-10000
            # ee_collision=-10

            # ee_outside_table = -20000

        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.9
        base_height_target = 1.
        max_contact_force = 1.  # forces above this value are penalized

        only_positive_rewards = False
        max_puck_vel = 0.4
        min_puck_ee_dist = 0.01
        max_vel_trunc_dist = 0.5
        min_dof_pos_done = 0.05  # rad >~ 0.1 deg
        min_dof_vel_done = 10  # rad >~ 0.1 deg
        min_ee_vel_diff = 0.001
        max_dof_qd=0.4
        max_ee_pd=0.015

        reset_on_success = True
        reset_on_fail = True

        adaptive_init = False
        adaptive_goal = True

class AirHockeyCfgPPO(LeggedRobotCfgPPO):
    num_actions = 3

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 200
        resume = False
        load_run = -1 #'Apr30_21-33-14_'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        max_iterations = 6000  # number of policy updates
        save_interval = 30  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # policy_class_name = 'ActorCriticRecurrent'
        # rnn_type = 'lstm'
        # rnn_hidden_size = 256
        # rnn_num_layers = 1
        # noise_std=1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.1
        # class high(LeggedRobotCfgPPO.algorithm):
        #     use_clipped_value_loss = True
        #     # desired_kl = 1
        #     # max_grad_norm = 0.1
        #     # learning_rate = 0.00001
        #     # num_mini_batches = 16
        #     entropy_coef = 0.001
        #     # schedule = 'fixed' # could be adaptive, fixed
        #     # std_coef=0.01
        #
        # class mid(LeggedRobotCfgPPO.algorithm):
        #     use_clipped_value_loss = True
        #     # desired_kl = 0.5
        #     # max_grad_norm = 0.1
        #     # learning_rate = 0.00001
        #     # num_mini_batches = 16
        #     entropy_coef = 0.001
        #     # schedule = 'fixed' # could be adaptive, fixed
        #
        #     # std_coef=0.01
        #
        # class low(LeggedRobotCfgPPO.algorithm):
        #     use_clipped_value_loss = True
        #     # desired_kl = 5e-4
        #     # max_grad_norm = 0.1
        #     # learning_rate = 0.00001
        #     # num_mini_batches = 16
        #     entropy_coef = 0.001
        #     # schedule = 'fixed' # could be adaptive, fixed
        #
        #     # std_coef=0.01

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 2
        # actor_hidden_dims = [512, 256]
        # critic_hidden_dims = [512, 256]
        # pass
        # class high(LeggedRobotCfgPPO.policy):
        #     num_actions = 2  # x,y,vel_x,vel_y
        #     num_obs = 18  # num_obs+mid_done
        #     num_steps = 200  # 50 high actions per episode
        #     num_steps_per_env = 20
        #     actor_hidden_dims = [256, 256]
        #     critic_hidden_dims = [256, 256]
        #     obs_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
        #     init_noise_std = 1
        #
        # class mid(LeggedRobotCfgPPO.policy):
        #     num_actions = 3  # q, qd for 3 joints
        #     num_obs = 13  # 6+high_actions+low_done q, qd for 3 joints
        #     num_steps = 20  # 5 mid action per high action
        #     num_steps_per_env = 200
        #     actor_hidden_dims = [128, 64]
        #     critic_hidden_dims = [128, 64]
        #     obs_idx = [6, 7, 8, 9, 10, 11, -1]
        #     init_noise_std = 1
        #
        # class low(LeggedRobotCfgPPO.policy):
        #     num_actions = 3  # q, qd for 3 joints
        #     num_obs = 13  # 6+mid_actions q, qd for 3 joints
        #     num_steps_per_env = 40
        #     num_steps = 1  # 20 low actions per mid action
        #     actor_hidden_dims = [128, 64]
        #     critic_hidden_dims = [128, 64]
        #     obs_idx = [6, 7, 8, 9, 10, 11]
        #     init_noise_std = 0.5
