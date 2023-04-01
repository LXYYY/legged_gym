from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AirHockeyCfg(LeggedRobotCfg):
    n_agents = 1

    def __init__(self):
        super().__init__()

    class env(LeggedRobotCfg.env):
        num_envs = 10
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


class AirHockeyCfgPPO(LeggedRobotCfgPPO):
    pass
